#include "tuning_manager.h"

#include <fstream>
#include <iostream>
#include <sstream>

// 解析自动调优结果 JSON 文件
namespace {

// 跳过 JSON 中的空白字符
void SkipWhitespace(std::istream& in) {
  while (in && std::isspace(in.peek())) {
    in.get();
  }
}

// 解析 JSON 字符串（带引号）
std::string ParseString(std::istream& in) {
  SkipWhitespace(in);
  if (in.get() != '"') return "";
  std::string result;
  while (in) {
    char c = in.get();
    if (c == '"') break;
    if (c == '\\') {
      c = in.get();  // 处理转义
    }
    result += c;
  }
  return result;
}

// 解析 JSON 数字
double ParseNumber(std::istream& in) {
  SkipWhitespace(in);
  double val = 0;
  in >> val;
  return val;
}

// 解析整数
int64_t ParseInteger(std::istream& in) {
  SkipWhitespace(in);
  int64_t val = 0;
  in >> val;
  return val;
}

// 跳过到指定字符
void SkipTo(std::istream& in, char target) {
  while (in && in.get() != target) {
  }
}

// 查找下一个键值对的键名
std::string NextKey(std::istream& in) {
  SkipWhitespace(in);
  if (in.peek() == '}' || in.peek() == ']') return "";
  if (in.peek() == ',') in.get();
  SkipWhitespace(in);
  if (in.peek() == '"') {
    auto key = ParseString(in);
    SkipTo(in, ':');
    return key;
  }
  return "";
}

}  // namespace

namespace infini::ops {

TuningManager& TuningManager::Instance() {
  static TuningManager instance;
  return instance;
}

void TuningManager::LoadTuningCache(const std::string& json_path) {
#ifndef WITH_TUNING
  // 编译时未启用调优，直接返回
  return;
#else
  std::lock_guard<std::mutex> lock(mutex_);

  // 记住路径：即便文件此刻不存在，之后 Record() 也会创建并写入它。
  json_path_ = json_path;
  // 编译期开启了 WITH_TUNING 即视为启用：允许在无缓存文件时现场测试并记录。
  enabled_ = true;

  std::ifstream file(json_path);
  if (!file.is_open()) {
    // 文件不存在或无法打开：首次运行的正常情况，以空缓存启动。
    return;
  }

  try {
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::istringstream in(buffer.str());

    // 解析根对象 { "version": 1, "entries": [...] }
    SkipTo(in, '{');
    std::string key;
    while ((key = NextKey(in)) != "") {
      if (key == "version") {
        int version = static_cast<int>(ParseInteger(in));
        if (version != 1) {
          std::cerr << "[TuningManager] Warning: tuning.json version "
                    << version << " not supported (expected 1)" << std::endl;
          return;
        }
      } else if (key == "entries") {
        // 解析 entries 数组
        SkipTo(in, '[');
        SkipWhitespace(in);
        while (in.peek() != ']') {
          // 解析每个 entry 对象
          SkipTo(in, '{');
          std::string op_name;
          Device::Type device = Device::Type::kCount;
          TuningSignature sig;
          std::size_t best_impl = 0;

          while ((key = NextKey(in)) != "") {
            if (key == "operator") {
              op_name = ParseString(in);
            } else if (key == "device") {
              std::string dev_str = ParseString(in);
              // 设备名映射（与 Device::StringFromType 对应）
              if (dev_str == "cpu") device = Device::Type::kCpu;
              else if (dev_str == "nvidia") device = Device::Type::kNvidia;
              else if (dev_str == "cambricon") device = Device::Type::kCambricon;
              else if (dev_str == "ascend") device = Device::Type::kAscend;
              else if (dev_str == "metax") device = Device::Type::kMetax;
              else if (dev_str == "moore") device = Device::Type::kMoore;
              else if (dev_str == "iluvatar") device = Device::Type::kIluvatar;
              else if (dev_str == "hygon") device = Device::Type::kHygon;
              // 其他设备类型可继续添加
            } else if (key == "signature") {
              // 解析签名对象 { "tensors": [...], "scalars": [...] }
              SkipTo(in, '{');
              while ((key = NextKey(in)) != "") {
                if (key == "tensors") {
                  // 解析张量数组
                  SkipTo(in, '[');
                  SkipWhitespace(in);
                  while (in.peek() != ']') {
                    SkipTo(in, '{');
                    TuningSignature::TensorSig tsig;
                    while ((key = NextKey(in)) != "") {
                      if (key == "shape") {
                        SkipTo(in, '[');
                        SkipWhitespace(in);
                        while (in.peek() != ']') {
                          tsig.shape.push_back(ParseInteger(in));
                          SkipWhitespace(in);
                          if (in.peek() == ',') in.get();
                          SkipWhitespace(in);
                        }
                        in.get();  // ']'
                      } else if (key == "dtype") {
                        tsig.dtype = static_cast<DataType>(ParseInteger(in));
                      } else {
                        SkipTo(in, ',');
                      }
                    }
                    sig.tensors.push_back(tsig);
                    SkipWhitespace(in);
                    if (in.peek() == ',') in.get();
                    SkipWhitespace(in);
                  }
                  in.get();  // ']'
                } else if (key == "scalars") {
                  // 解析标量数组
                  SkipTo(in, '[');
                  SkipWhitespace(in);
                  while (in.peek() != ']') {
                    sig.scalars.push_back(ParseNumber(in));
                    SkipWhitespace(in);
                    if (in.peek() == ',') in.get();
                    SkipWhitespace(in);
                  }
                  in.get();  // ']'
                } else {
                  SkipTo(in, ',');
                }
              }
            } else if (key == "best_implementation") {
              best_impl = static_cast<std::size_t>(ParseInteger(in));
            } else if (key == "metadata") {
              // 跳过 metadata（不参与查找）
              int depth = 0;
              SkipWhitespace(in);
              char c = in.get();
              if (c == '{') depth = 1;
              while (depth > 0 && in) {
                c = in.get();
                if (c == '{') depth++;
                else if (c == '}') depth--;
              }
            } else {
              SkipTo(in, ',');
            }
          }

          // 将解析的条目加入缓存
          if (!op_name.empty() && device != Device::Type::kCount) {
            CacheKey cache_key{op_name, device, sig};
            cache_[cache_key] = best_impl;
          }

          SkipWhitespace(in);
          if (in.peek() == ',') in.get();
          SkipWhitespace(in);
        }
      } else {
        SkipTo(in, ',');
      }
    }

    std::cout << "[TuningManager] Loaded " << cache_.size()
              << " tuning entries from " << json_path << std::endl;

  } catch (...) {
    // 解析失败：丢弃可能残缺的记录，但保持 enabled_，
    // 让运行期仍能现场测试并用正确内容覆盖损坏的文件。
    std::cerr << "[TuningManager] Warning: failed to parse " << json_path
              << ", starting with an empty cache" << std::endl;
    cache_.clear();
  }
#endif
}

std::optional<std::size_t> TuningManager::Lookup(
    const std::string& operator_name, Device::Type device,
    const TuningSignature& signature) const {
#ifndef WITH_TUNING
  // 编译时未启用调优，直接返回空
  return std::nullopt;
#else
  if (!enabled_) return std::nullopt;

  std::lock_guard<std::mutex> lock(mutex_);
  CacheKey key{operator_name, device, signature};
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    return it->second;
  }
  return std::nullopt;
#endif
}

void TuningManager::Record(const std::string& operator_name,
                           Device::Type device,
                           const TuningSignature& signature,
                           std::size_t best_index) {
#ifndef WITH_TUNING
  // 编译时未启用调优，什么也不做
  (void)operator_name;
  (void)device;
  (void)signature;
  (void)best_index;
  return;
#else
  if (!enabled_) return;

  std::lock_guard<std::mutex> lock(mutex_);
  CacheKey key{operator_name, device, signature};
  cache_[key] = best_index;
  // 落盘
  FlushToDiskLocked();
#endif
}

void TuningManager::FlushToDiskLocked() const {
#ifdef WITH_TUNING
  std::ofstream out(json_path_, std::ios::trunc);
  if (!out.is_open()) {
    std::cerr << "[TuningManager] Warning: cannot write tuning cache to "
              << json_path_ << std::endl;
    return;
  }

  // 手写 JSON 序列化，结构与 LoadTuningCache 的解析器完全对应。
  out << "{\n";
  out << "  \"version\": 1,\n";
  out << "  \"entries\": [\n";

  std::size_t entry_index = 0;
  for (const auto& [key, best_impl] : cache_) {
    out << "    {\n";
    out << "      \"operator\": \"" << key.operator_name << "\",\n";
    out << "      \"device\": \""
        << Device::StringFromType(key.device) << "\",\n";
    out << "      \"signature\": {\n";

    // tensors 数组
    out << "        \"tensors\": [";
    for (std::size_t i = 0; i < key.signature.tensors.size(); ++i) {
      const auto& t = key.signature.tensors[i];
      out << (i == 0 ? "\n" : ",\n");
      out << "          {\"shape\": [";
      for (std::size_t d = 0; d < t.shape.size(); ++d) {
        out << (d == 0 ? "" : ", ") << t.shape[d];
      }
      out << "], \"dtype\": " << static_cast<int>(t.dtype) << "}";
    }
    out << (key.signature.tensors.empty() ? "" : "\n        ") << "],\n";

    // scalars 数组
    out << "        \"scalars\": [";
    for (std::size_t i = 0; i < key.signature.scalars.size(); ++i) {
      out << (i == 0 ? "" : ", ") << key.signature.scalars[i];
    }
    out << "]\n";

    out << "      },\n";
    out << "      \"best_implementation\": " << best_impl << "\n";
    out << "    }" << (++entry_index < cache_.size() ? "," : "") << "\n";
  }

  out << "  ]\n";
  out << "}\n";
#endif
}

}  // namespace infini::ops