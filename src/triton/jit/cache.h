#ifndef INFINI_OPS_TRITON_JIT_CACHE_H_
#define INFINI_OPS_TRITON_JIT_CACHE_H_

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "jit.h"

namespace infini::ops {

// ---- file helpers ----

inline bool file_exists(const char* path) {
  FILE* f = fopen(path, "rb");
  if (f != nullptr) {
    fclose(f);
    return true;
  }
  return false;
}

inline std::string read_file(const char* path) {
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

inline bool cache_complete(const std::string& cubin_path,
                           const std::string& meta_path) {
  return file_exists(cubin_path.c_str()) && file_exists(meta_path.c_str());
}

// ---- json field extraction ----

inline int json_get_int(const std::string& json, const char* key,
                        int fallback = 0) {
  std::string pat = std::string("\"") + key + "\":";
  auto pos = json.find(pat);
  if (pos == std::string::npos) return fallback;
  pos += pat.size();
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
  return std::atoi(json.c_str() + pos);
}

inline std::string json_get_string(const std::string& json, const char* key,
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

// ---- kernel cache ----

inline std::string generate_desc(const char* op, const char* sig,
                                 unsigned num_warps, unsigned num_stages,
                                 int arch) {
  return std::string(op) + "|" + sig + "|" + std::to_string(num_warps) + "|" +
         std::to_string(num_stages) + "|sm" + std::to_string(arch);
}

inline std::string cache_mem_key(const char* op_name, const char* signature_str,
                                 unsigned num_warps, unsigned num_stages,
                                 int arch, int dev_id) {
  return generate_desc(op_name, signature_str, num_warps, num_stages, arch) +
         "|dev" + std::to_string(dev_id);
}

inline std::string cache_file_key(const char* op_name,
                                  const char* signature_str, unsigned num_warps,
                                  unsigned num_stages, int arch) {
  return std::to_string(std::hash<std::string>{}(
      generate_desc(op_name, signature_str, num_warps, num_stages, arch)));
}

struct kernel_cache_entry_t {
  void* func;
  unsigned shared;
};

struct kernel_cache_t {
  std::mutex mutex;
  std::unordered_map<std::string, kernel_cache_entry_t> map;
};

inline kernel_cache_t& kernel_cache() {
  static kernel_cache_t c;
  return c;
}

inline bool kernel_cache_lookup(const std::string& key,
                                kernel_cache_entry_t* out) {
  auto& c = kernel_cache();
  std::lock_guard<std::mutex> lk(c.mutex);
  auto it = c.map.find(key);
  if (it == c.map.end()) return false;
  *out = it->second;
  return true;
}

inline void kernel_cache_insert(const std::string& key,
                                kernel_cache_entry_t entry) {
  auto& c = kernel_cache();
  std::lock_guard<std::mutex> lk(c.mutex);
  c.map[key] = entry;
}

struct cache_query_result_t {
  bool mem_hit;
  void* func;
  unsigned shared;
  std::string out_prefix;
  std::string mem_key;
};

inline cache_query_result_t cache_query(const char* op, const char* sig,
                                        unsigned num_warps, unsigned num_stages,
                                        int arch, int dev_id) {
  auto mem_key = cache_mem_key(op, sig, num_warps, num_stages, arch, dev_id);
  kernel_cache_entry_t entry;
  if (kernel_cache_lookup(mem_key, &entry))
    return {true, entry.func, entry.shared, "", mem_key};
  auto desc = generate_desc(op, sig, num_warps, num_stages, arch);
  return {false, nullptr, 0,
          std::string(TRITON_JIT_CACHE_DIR) + "/" +
              std::to_string(std::hash<std::string>{}(desc)),
          mem_key};
}

struct autotune_cache_t {
  std::mutex mutex;
  std::unordered_map<std::string, config_t> map;
};

inline autotune_cache_t& autotune_cache() {
  static autotune_cache_t c;
  return c;
}

inline std::string autotune_cache_file_path(const std::string& key) {
  return std::string(TRITON_JIT_CACHE_DIR) + "/" +
         std::to_string(std::hash<std::string>{}(key)) + ".autotune";
}

inline std::string serialize_config(const config_t& config) {
  std::string s = std::to_string(config.num_warps) + " " +
                  std::to_string(config.num_stages);
  for (const auto& [name, val] : config.constexprs)
    s += "\n" + name + " " + std::to_string(val);
  return s;
}

inline bool deserialize_config(const std::string& content, config_t* out) {
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

inline bool autotune_cache_lookup(const std::string& key, config_t* out) {
  auto& c = autotune_cache();
  std::lock_guard<std::mutex> lk(c.mutex);
  auto it = c.map.find(key);
  if (it != c.map.end()) {
    *out = it->second;
    return true;
  }
  std::string path = autotune_cache_file_path(key);
  if (file_exists(path.c_str())) {
    config_t parsed;
    if (deserialize_config(read_file(path.c_str()), &parsed)) {
      c.map[key] = parsed;
      *out = parsed;
      return true;
    }
  }
  return false;
}

inline void autotune_cache_insert(const std::string& key,
                                  const config_t& config) {
  auto& c = autotune_cache();
  std::lock_guard<std::mutex> lk(c.mutex);
  c.map[key] = config;
  std::string path = autotune_cache_file_path(key);
  std::string content = serialize_config(config);
  FILE* f = fopen(path.c_str(), "w");
  if (f) {
    fwrite(content.data(), 1, content.size(), f);
    fclose(f);
  }
}

}  // namespace infini::ops

#endif
