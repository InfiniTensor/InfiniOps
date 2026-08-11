#include "tuning.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace {

void SkipWhitespace(std::istream& in) {
  while (in && std::isspace(in.peek())) {
    in.get();
  }
}

std::string ParseString(std::istream& in) {
  SkipWhitespace(in);
  if (in.get() != '"') return "";
  std::string result;
  while (in) {
    char c = in.get();
    if (c == '"') break;
    if (c == '\\') {
      c = in.get();
    }
    result += c;
  }
  return result;
}

double ParseNumber(std::istream& in) {
  SkipWhitespace(in);
  double val = 0;
  in >> val;
  return val;
}

int64_t ParseInteger(std::istream& in) {
  SkipWhitespace(in);
  int64_t val = 0;
  in >> val;
  return val;
}

void SkipTo(std::istream& in, char target) {
  while (in && in.get() != target) {
  }
}

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
  std::lock_guard<std::mutex> lock(mutex_);

  json_path_ = json_path;
  enabled_ = true;

  std::ifstream file(json_path);
  if (!file.is_open()) {
    return;
  }

  try {
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::istringstream in(buffer.str());

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
        SkipTo(in, '[');
        SkipWhitespace(in);
        while (in && in.peek() != ']') {
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
              if (dev_str == "cpu")
                device = Device::Type::kCpu;
              else if (dev_str == "nvidia")
                device = Device::Type::kNvidia;
              else if (dev_str == "cambricon")
                device = Device::Type::kCambricon;
              else if (dev_str == "ascend")
                device = Device::Type::kAscend;
              else if (dev_str == "metax")
                device = Device::Type::kMetax;
              else if (dev_str == "moore")
                device = Device::Type::kMoore;
              else if (dev_str == "iluvatar")
                device = Device::Type::kIluvatar;
              else if (dev_str == "hygon")
                device = Device::Type::kHygon;
            } else if (key == "signature") {
              SkipTo(in, '{');
              while ((key = NextKey(in)) != "") {
                if (key == "tensors") {
                  SkipTo(in, '[');
                  SkipWhitespace(in);
                  while (in && in.peek() != ']') {
                    SkipTo(in, '{');
                    TuningSignature::TensorSig tsig;
                    while ((key = NextKey(in)) != "") {
                      if (key == "shape") {
                        SkipTo(in, '[');
                        SkipWhitespace(in);
                        while (in && in.peek() != ']') {
                          tsig.shape.push_back(ParseInteger(in));
                          SkipWhitespace(in);
                          if (in.peek() == ',') in.get();
                          SkipWhitespace(in);
                        }
                        if (in.peek() == ']') in.get();
                      } else if (key == "dtype") {
                        tsig.dtype = static_cast<DataType>(ParseInteger(in));
                      } else {
                        SkipTo(in, ',');
                      }
                    }
                    if (in.peek() == '}') in.get();
                    sig.tensors.push_back(tsig);
                    SkipWhitespace(in);
                    if (in.peek() == ',') in.get();
                    SkipWhitespace(in);
                  }
                  if (in.peek() == ']') in.get();
                } else if (key == "scalars") {
                  SkipTo(in, '[');
                  SkipWhitespace(in);
                  while (in && in.peek() != ']') {
                    sig.scalars.push_back(ParseNumber(in));
                    SkipWhitespace(in);
                    if (in.peek() == ',') in.get();
                    SkipWhitespace(in);
                  }
                  if (in.peek() == ']') in.get();
                } else {
                  SkipTo(in, ',');
                }
              }
              if (in.peek() == '}') in.get();
            } else if (key == "best_implementation") {
              best_impl = static_cast<std::size_t>(ParseInteger(in));
            } else if (key == "metadata") {
              int depth = 0;
              SkipWhitespace(in);
              char c = in.get();
              if (c == '{') depth = 1;
              while (depth > 0 && in) {
                c = in.get();
                if (c == '{')
                  depth++;
                else if (c == '}')
                  depth--;
              }
            } else {
              SkipTo(in, ',');
            }
          }

          if (in.peek() == '}') in.get();

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
    std::cerr << "[TuningManager] Warning: failed to parse " << json_path
              << ", starting with an empty cache" << std::endl;
    cache_.clear();
  }
}

std::optional<std::size_t> TuningManager::Lookup(
    const std::string& operator_name, Device::Type device,
    const TuningSignature& signature) const {
  if (!enabled_) return std::nullopt;

  std::lock_guard<std::mutex> lock(mutex_);
  CacheKey key{operator_name, device, signature};
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    return it->second;
  }
  return std::nullopt;
}

void TuningManager::Record(const std::string& operator_name,
                           Device::Type device,
                           const TuningSignature& signature,
                           std::size_t best_index) {
  if (!enabled_) return;

  std::lock_guard<std::mutex> lock(mutex_);
  CacheKey key{operator_name, device, signature};
  cache_[key] = best_index;
  FlushToDiskLocked();
}

void TuningManager::FlushToDiskLocked() const {
  std::ofstream out(json_path_, std::ios::trunc);
  if (!out.is_open()) {
    std::cerr << "[TuningManager] Warning: cannot write tuning cache to "
              << json_path_ << std::endl;
    return;
  }

  out << "{\n";
  out << "  \"version\": 1,\n";
  out << "  \"entries\": [\n";

  std::size_t entry_index = 0;
  for (const auto& [key, best_impl] : cache_) {
    out << "    {\n";
    out << "      \"operator\": \"" << key.operator_name << "\",\n";
    out << "      \"device\": \"" << Device::StringFromType(key.device)
        << "\",\n";
    out << "      \"signature\": {\n";

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
}

}  // namespace infini::ops
