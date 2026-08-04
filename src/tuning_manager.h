#ifndef INFINI_OPS_TUNING_MANAGER_H_
#define INFINI_OPS_TUNING_MANAGER_H_

#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "device.h"
#include "tuning_signature.h"

namespace infini::ops {

// 调优管理器：单例模式，负责查询、记录、持久化调优缓存。
//
//   - 启动时若存在 tuning.json，则加载已有记录（可选，没有也没关系）；
//   - 运行期算子首次遇到某形状时，由调用方现场基准测试并调用 Record() 写入；
//   - 之后相同形状直接 Lookup() 命中，零额外开销。
//
// 线程安全：运行期存在并发的查询与写入，故用互斥锁保护缓存与落盘。
class TuningManager {
 public:
  // 获取单例实例
  static TuningManager& Instance();

  // 从 JSON 文件加载调优缓存（若文件存在）。
  // 参数：json_path - tuning.json 的路径。
  void LoadTuningCache(const std::string& json_path);

  // 查询最优实现索引。
  // 参数：
  //   operator_name - 算子名称（如 "RmsNorm"）
  //   device - 设备类型（如 Device::Type::kNvidia）
  //   signature - 调优签名（形状+类型+标量参数）
  // 返回：命中则返回最优索引，否则返回 std::nullopt。
  std::optional<std::size_t> Lookup(const std::string& operator_name,
                                    Device::Type device,
                                    const TuningSignature& signature) const;

  // 记录一条调优结果：更新内存缓存，并立即把整个缓存写回 tuning.json。
  // 参数：
  //   operator_name - 算子名称
  //   device - 设备类型
  //   signature - 调优签名
  //   best_index - 现场基准测试选出的最快实现索引
  void Record(const std::string& operator_name, Device::Type device,
              const TuningSignature& signature, std::size_t best_index);

  // 检查是否启用调优。编译时开启 WITH_TUNING 即为 true（无论有无缓存文件）。
  bool IsEnabled() const { return enabled_; }

 private:
  TuningManager() = default;

  // 禁止拷贝和赋值（单例模式）
  TuningManager(const TuningManager&) = delete;
  TuningManager& operator=(const TuningManager&) = delete;

  // 调优缓存的键：算子名 + 设备 + 签名
  struct CacheKey {
    std::string operator_name;
    Device::Type device;
    TuningSignature signature;

    bool operator==(const CacheKey& other) const {
      return operator_name == other.operator_name && device == other.device &&
             signature == other.signature;
    }
  };

  // 哈希函数
  struct CacheKeyHash {
    std::size_t operator()(const CacheKey& key) const {
      std::size_t h = std::hash<std::string>{}(key.operator_name);
      h ^= std::hash<int>{}(static_cast<int>(key.device)) + 0x9e3779b9 +
           (h << 6) + (h >> 2);
      h ^= key.signature.Hash() + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  // 把当前内存缓存序列化写回 json_path_（调用前须已持有 mutex_）。
  void FlushToDiskLocked() const;

  // 调优缓存：(算子名, 设备, 签名) -> 最优实现索引
  std::unordered_map<CacheKey, std::size_t, CacheKeyHash> cache_;

  // 是否已启用调优（WITH_TUNING 开启即为 true）
  bool enabled_{false};

  // tuning.json 的路径，Record() 据此落盘
  std::string json_path_{"tuning.json"};

  // 保护 cache_ 与落盘操作（运行期读写并发）
  mutable std::mutex mutex_;
};

}  // namespace infini::ops

#endif