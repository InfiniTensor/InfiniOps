# Linux CPU 安装消费者验证

此通道是 P0.2 的首个完整配方，验证 native CPU Add/Gemm 的已安装 C++/Python 入口。新通道不更改现有硬件 CI、算子、接口、默认选择或运行逻辑。其他平台和算子仍保留，尚未通过本配方验证，不得据此宣称它们已经通过。

## 固定输入

- Linux x86_64，官方 Python 3.12.10 Bookworm 镜像，固定 OCI digest；镜像内的编译器和标准库保持配套。
- InfiniRT 提交 `cd1d48258f0a6b8d9f66111d5d6cb1664205e78d`，CPU Release 构建。
- [requirements.in](requirements.in) 固定直接依赖，[requirements.lock](requirements.lock) 固定解析后的传递依赖及下载 hash。安装使用 `--require-hashes --only-binary=:all:`。
- 当前提交的 InfiniOps，固定 `add,gemm` 清单，CPU 开启，自动探测与 Torch/linked/DSL provider 关闭。Torch 仅作为 Python Tensor 输入与参考检查工具。
- checkout/upload-artifact action 固定提交；Hosted runner 提供容器执行，实际编译发生在固定镜像内。

更新依赖需要独立审查 lock diff，重新完成实际构建和消费者检查，不能跟随 RT HEAD 或自动放宽版本。源树审查基线覆盖本通道的 recipe、lock、消费者及执行器输入。

## 执行链路与拒绝条件

[Installed CPU Consumers](../../.github/workflows/ci-installed-consumers.yml) 工作流依次安装锁定依赖、运行故障路径测试、构建安装 RT、构建一个 InfiniOps wheel 并安装，再运行消费者。C++ 和 Python 使用同一份 wheel 里的库、头和 CMake package；不额外从源码复制头或库来修复安装缺失。

在具备上述配套安装的 Linux 环境中，可单独运行：

```text
python scripts/verify_installed_consumers.py --prefix /installed/site-packages/infini --rt-prefix /installed/rt --output /new/consumer-evidence
```

安装 prefix 和证据目录必须在源码树外。证据目录必须不存在，避免复用旧报告；验证失败保留 `report.json` 与已产生的命令日志，不晋级为通过。

执行器从仓库原样复制 `test_cpp_api.py`，不复制 `conftest.py`，只选择安装消费者、JSON 调用追踪和 returning API 三个既有测试。它们以独立 Python 进程从源码树外运行，关闭第三方 pytest 插件自动加载并清除 Python 路径覆盖。必需测试缺失、重复、零执行、skip、failure 或 error 都阻止通过；原测试本身的 skip 行为不改。

另一个仅依赖 `find_package(InfiniOps)` 的消费者输出 Add/Gemm 的实际 active/default 映射，预期均为 slot 0。独立 Python 消费者使用旧 `infini.ops` 入口检查 Add 默认调用、alpha 重载与 Gemm 固定 golden，记录实际模块位置、导出名和已加载库位置。

证据包含命令与返回码、core/RT 身份、依赖清单、编译器版本、验证输入 hash、安装文件 hash、默认/active 映射、导出符号和动态依赖。wheel 中重复安装的 `libinfiniops.so` 必须内容一致。GitHub artifact 名包含提交 SHA；构建本身失败时以该提交的 step 日志为证据，不把缺失消费者报告解释为成功。

## 验证层次

```text
python -m unittest discover -s tests -p test_installed_consumer_guard.py -v
```

标准库单测只验证故障分类与报告机制，不模拟真实算子成功；实际构建、C++ 数值调用、Python 数值调用和打包效果必须由上述 Linux 通道给出结果。本配方不证明完整业务、全部外部框架、其他硬件或性能基线已经验收，也不代表 P0 整体完成。
