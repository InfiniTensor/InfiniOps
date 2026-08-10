#ifndef INFINI_OPS_TRITON_JIT_CACHE_H_
#define INFINI_OPS_TRITON_JIT_CACHE_H_

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "base.h"

namespace infini::ops {

// ---- file helpers ----

inline bool FileExists(const char* path) {
  FILE* f = fopen(path, "rb");
  if (f != nullptr) {
    fclose(f);
    return true;
  }
  return false;
}

inline std::string ReadFile(const char* path) {
  FILE* f = fopen(path, "rb");
  if (f == nullptr) return {};
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  if (sz < 0) {
    fclose(f);
    return {};
  }
  fseek(f, 0, SEEK_SET);
  std::string buf(static_cast<size_t>(sz), '\0');
  size_t nread = fread(buf.data(), 1, static_cast<size_t>(sz), f);
  fclose(f);
  buf.resize(nread);
  return buf;
}

// ---- JSON field extraction ----

inline int JsonGetInt(const std::string& json, const char* key,
                      int fallback = 0) {
  std::string pat = std::string("\"") + key + "\":";
  auto pos = json.find(pat);
  if (pos == std::string::npos) return fallback;
  pos += pat.size();
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
  return std::atoi(json.c_str() + pos);
}

inline std::string JsonGetString(const std::string& json, const char* key,
                                 const char* fallback) {
  std::string pat = std::string("\"") + key + "\":";
  auto pos = json.find(pat);
  if (pos == std::string::npos) return fallback;
  pos += pat.size();
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
  if (pos >= json.size() || json[pos] != '"') return fallback;
  pos++;
  auto end = json.find('"', pos);
  if (end == std::string::npos) return fallback;
  return json.substr(pos, end - pos);
}

// ---- key generation ----

inline std::string GenerateDesc(const char* op, const char* sig,
                                unsigned num_warps, unsigned num_stages,
                                int arch) {
  return std::string(op) + "|" + sig + "|" + std::to_string(num_warps) + "|" +
         std::to_string(num_stages) + "|sm" + std::to_string(arch);
}

inline std::string CacheMemKey(const char* op_name, const char* signature_str,
                               unsigned num_warps, unsigned num_stages,
                               int arch, int dev_id) {
  return GenerateDesc(op_name, signature_str, num_warps, num_stages, arch) +
         "|dev" + std::to_string(dev_id);
}

inline std::string CacheFileKey(const char* op_name, const char* signature_str,
                                unsigned num_warps, unsigned num_stages,
                                int arch) {
  return std::to_string(std::hash<std::string>{}(
      GenerateDesc(op_name, signature_str, num_warps, num_stages, arch)));
}

// ---- artifact reader ----

inline bool ReadArtifacts(const std::string& out_prefix, KernelMeta* meta,
                          std::string* binary_data) {
  std::string meta_path = out_prefix + ".json";
  std::string meta_json = ReadFile(meta_path.c_str());
  if (meta_json.empty()) return false;

  meta->name = JsonGetString(meta_json, "name", "");
  meta->binary_ext = JsonGetString(meta_json, "binary_ext", "");
  if (meta->binary_ext.empty()) meta->binary_ext = "cubin";
  meta->shared = JsonGetInt(meta_json, "shared");
  meta->num_warps = JsonGetInt(meta_json, "num_warps");
  meta->global_scratch_size = JsonGetInt(meta_json, "global_scratch_size");
  meta->profile_scratch_size = JsonGetInt(meta_json, "profile_scratch_size");

  std::string binary_path = out_prefix + "." + meta->binary_ext;
  *binary_data = ReadFile(binary_path.c_str());
  return !binary_data->empty();
}

// ---- kernel cache ----

template <Device::Type kDev>
struct KernelCacheEntry {
  typename Driver<kDev>::Function func;

  unsigned shared;
};

template <Device::Type kDev>
struct KernelCache {
  std::mutex mutex;

  std::unordered_map<std::string, KernelCacheEntry<kDev>> map;
};

template <Device::Type kDev>
KernelCache<kDev>& GetKernelCache() {
  static KernelCache<kDev> c;
  return c;
}

template <Device::Type kDev>
bool KernelCacheLookup(const std::string& key, KernelCacheEntry<kDev>* out) {
  auto& c = GetKernelCache<kDev>();
  std::lock_guard<std::mutex> lk(c.mutex);
  auto it = c.map.find(key);
  if (it == c.map.end()) return false;
  *out = it->second;
  return true;
}

template <Device::Type kDev>
void KernelCacheInsert(const std::string& key, KernelCacheEntry<kDev> entry) {
  auto& c = GetKernelCache<kDev>();
  std::lock_guard<std::mutex> lk(c.mutex);
  c.map[key] = entry;
}

template <Device::Type kDev>
struct CacheQueryResult {
  bool mem_hit;

  typename Driver<kDev>::Function func;

  unsigned shared;

  std::string out_prefix;

  std::string mem_key;
};

template <Device::Type kDev>
CacheQueryResult<kDev> CacheQuery(const char* op, const char* sig,
                                  unsigned num_warps, unsigned num_stages,
                                  int arch, int dev_id) {
  auto mem_key = CacheMemKey(op, sig, num_warps, num_stages, arch, dev_id);
  KernelCacheEntry<kDev> entry;
  if (KernelCacheLookup<kDev>(mem_key, &entry))
    return {true, entry.func, entry.shared, "", mem_key};
  auto desc = GenerateDesc(op, sig, num_warps, num_stages, arch);
  return {false, nullptr, 0,
          std::string(TRITON_JIT_CACHE_DIR) + "/" +
              std::to_string(std::hash<std::string>{}(desc)),
          mem_key};
}

// ---- autotune cache ----

struct AutotuneCache {
  std::mutex mutex;

  std::unordered_map<std::string, JitConfig> map;
};

inline AutotuneCache& GetAutotuneCache() {
  static AutotuneCache c;
  return c;
}

inline std::string AutotuneCacheFilePath(const std::string& key) {
  return std::string{TRITON_JIT_CACHE_DIR} + "/" +
         std::to_string(std::hash<std::string>{}(key)) + ".autotune";
}

inline std::string SerializeConfig(const JitConfig& config) {
  std::string s = std::to_string(config.num_warps) + " " +
                  std::to_string(config.num_stages);
  for (const auto& [name, val] : config.constexprs)
    s += "\n" + name + " " + std::to_string(val);
  return s;
}

inline bool DeserializeConfig(const std::string& content, JitConfig* out) {
  std::istringstream iss(content);
  std::string line;
  if (!std::getline(iss, line)) return false;
  std::istringstream head(line);
  if (!(head >> out->num_warps >> out->num_stages)) return false;
  out->constexprs.clear();
  while (std::getline(iss, line)) {
    std::istringstream ls(line);
    std::string name;
    int val;
    if (ls >> name >> val) out->constexprs.push_back({name, val});
  }
  return true;
}

inline bool AutotuneCacheLookup(const std::string& key, JitConfig* out) {
  auto& c = GetAutotuneCache();
  std::lock_guard<std::mutex> lk(c.mutex);
  auto it = c.map.find(key);
  if (it != c.map.end()) {
    *out = it->second;
    return true;
  }
  std::string path = AutotuneCacheFilePath(key);
  if (FileExists(path.c_str())) {
    JitConfig parsed;
    if (DeserializeConfig(ReadFile(path.c_str()), &parsed)) {
      c.map[key] = parsed;
      *out = parsed;
      return true;
    }
  }
  return false;
}

inline void AutotuneCacheInsert(const std::string& key,
                                const JitConfig& config) {
  auto& c = GetAutotuneCache();
  std::lock_guard<std::mutex> lk(c.mutex);
  c.map[key] = config;
  std::string path = AutotuneCacheFilePath(key);
  std::string content = SerializeConfig(config);
  FILE* f = fopen(path.c_str(), "w");
  if (f) {
    fwrite(content.data(), 1, content.size(), f);
    fclose(f);
  }
}

}  // namespace infini::ops

#endif
