# InfiniOps Review 指南

本指南定义 reviewer 在 InfiniOps pull request 中应关注的问题。除非子目录中的 `REVIEW.md` 对其所在子树进一步收窄或扩展规则，本指南递归适用于本目录下的所有文件。

本指南同时供人工 review 和 AI review 使用。它是建议性指南，不能替代 maintainer 判断、必需的 CI，或项目要求的审批。基础编码、格式、commit 和测试约定以 `CONTRIBUTING.md` 为准；本指南重点覆盖 CI 和机械规则无法可靠判断的问题。

## Review 原则

- 关注 CI 和自动格式化工具无法可靠发现的问题：设计、正确性、兼容性、平台影响、测试充分性、benchmark 有效性、可维护性和安全性。
- 不要把 review 时间花在重复报告 CI 已经可见的格式化、lint、普通 build 或测试失败上。
- 只报告可行动的问题、风险和具体替代方案。避免赞美式评论、没有维护收益的风格意见，以及只是复述 diff 的评论。
- 同时 review 变更后的设计及其调用方。先调查，不要猜测；证据不足时应说明缺失的证据，而不是虚构行为。
- 将发现分为 **Blocking** 和 **Non-blocking**。Blocking 表示在问题修复或 maintainer 明确接受风险之前，PR 不应合并。
- 引用精确的 `path:line` 位置，必要时应给出有代表性的小例子或失败输入。优先给出建议改法，而不是模糊偏好。
- rename 和 delete 的旧路径、移动后的新路径都要检查。检查周边代码和类似实现，判断变更是否符合局部模式。
- 发现同类问题时，应检查相关文件或子系统中是否还有重复出现；同时仍要求修复范围保持在 PR scope 内。

## 必要性、范围和可维护性

- “如无必要，勿增实体。” 与目标变更无关的空行、重命名、include、公共接口调整、build 逻辑和开发辅助代码不应混入 PR。
- 公共基础接口、核心 dispatcher、type map、runtime wrapper 和重要 header 的修改必须有明确理由。能用局部私有 helper 解决时，不应扩大公共 API 或暴露内部状态。
- 新增文件、类、函数、成员、宏、配置项和测试工具前必须说明必要性。确认无用时应删除，而不是留在仓库中等待未来使用。
- 抽象必须有多个真实使用者或清晰的后继计划。避免为了一个调用点引入层层转发；也避免为了省一处代码引入需要长期维护的通用层。
- CUDA-compatible operator 的实现层级必须清晰：通用 kernel、索引计算、shape/layout/overlap 校验、错误处理和可跨厂商复用的数值逻辑应放在 `src/native/cuda/ops/{operator}` 并以 `Backend` 模板化；vendor 目录只保留真正的 SDK intrinsic、runtime 扩展、数值语义或调度策略差异和薄 wrapper。不要因为 PR 先服务一个厂商，就把可复用实现放进该厂商目录。
- 共享 CUDA-compatible template 中不应用 `if constexpr (Backend::kDeviceType == ...)` 堆放 vendor-only fast path。应下沉到 vendor specialization，或抽出 backend policy/template 参数；默认路径和其他厂商的编译、行为和维护成本不应受无关特化影响。
- C++ 中能用 `constexpr`、`const`、函数或类型系统表达的行为，不应使用宏。必须使用宏时应限制作用域并检查重复定义。
- device 查询、stream 转换、内存分配/拷贝、context 和错误码处理不得在 operator 中直接依赖具体厂商 C API（如 `xxGetDevice`、`xxStream_t`）。CUDA-compatible 实现应通过 InfiniRT `Runtime` / Backend 抽象访问（如 `Backend::GetDevice`、`Backend::Stream`、`Backend::Memcpy`）；vendor SDK 类型应留在 runtime adapter 层。
- 开发期工具、临时缓存策略或未启用的后端能力应单独 PR，说明设计、生命周期和默认行为，不应夹在 operator PR 中。
- 删除或合并重复定义时，应保留一份清晰的声明或定义，并检查 include 依赖方向，避免上层文件依赖下层实现细节。

## PR 元信息和平台范围

- PR title 和 commit 必须遵循 Conventional Commits。branch type 应与 PR type 匹配；breaking change 必须带 `!` 或 `BREAKING CHANGE` footer。
- PR template 的每个 section 都必须填写。确实不适用的 section 应说明为什么是 `N/A`。
- Summary 必须同时说明改了什么、为什么改，并引用相关文件或 issue。
- **Platforms Affected** 必须包含所有代码、生成绑定、build 选择、dispatch 行为或可观察语义可能变化的 backend，而不只是 diff 直接可见修改的目录。
- 共享 CUDA-compatible template、dispatcher 逻辑、wrapper、header、type map、build script 和公共 test helper 可能影响多个 vendor backend。应把更大的平台集合视为受影响。
- pending CI 绝不能写成 passed。未测试的受影响平台必须写明原因，并转给能够验证它的 reviewer 或 owner。
- 平台差异必须显式建模。某平台要求 shape、dtype、layout 或参数一致时，应在 base contract、`supports()` 或 debug 检查中表达，不能等到 native path 内静默解释。

## Operator 契约和 API 兼容性

- Operator 参数遵循现有顺序：input 在前，attribute 在中间，output 在后。命名优先遵循 PyTorch 约定，其次是 ONNX，再其次是 CUDA API。
- 新增或修改的 operator 必须选定并文档化上游/API 目标。不得在只实现了上游 API 最方便的默认子集时宣称与其兼容。
- 上游对齐必须逐项检查参数名、参数顺序、默认值、可选参数、dtype、shape、返回值、返回形态和错误行为。常见遗漏包括 `alpha`、`beta`、`bias`、`tau`、`decimals`、`stable`、`dtype`、reduction、weight、ignore index 和无 `dim` 形式。
- 不应为了避开一个参数而拆出近似 operator。优先扩展现有 operator，使一个公开契约能表达相关变体；确实需要拆分时，必须给出独立、稳定、语义完整的名称。
- 不应使用 `input2`、`_grad_input`、`_forward_output`、`_dim`、`_scalar` 这类用参数名、输出名或实现细节拼接的后缀来扩展 operator 名称。
- Operator 成员必须完整保存后续调用所需的状态。构造函数、成员声明顺序、初始化列表、`operator()` 参数和 wrapper 生成的调用顺序应来自同一个契约。
- `std::optional`、默认值和参数缺省行为应与上游目标对齐。上游可选参数进入实现后，不能只保存在构造签名中而丢失在调用路径中。
- 如果有意不支持某个上游能力，必须明确命名受限契约，并拒绝不支持的合法输入，而不是接受它们并产生误导性结果。
- forward 契约和 backward 契约必须一致。backward 实现必须保留 forward 的所有 attribute，应用与上游一致的梯度规则，并将梯度归约回每个原始输入的 shape。
- 保留 broadcast 语义。支持 broadcast 的 forward operator 必须有处理 right-aligned broadcasting 的 backward，并把梯度归约回每个输入 shape；不得发明项目私有的 broadcast 规则。
- 公开名称、enum、错误信息、涉及 ABI 的签名以及 serialization 兼容行为应保持稳定，除非 PR 有意文档化并 review 一次迁移方案。switch/case 和映射表的顺序应与基础 enum 定义保持一致。
- 不要给 deprecated operator 增加新行为或新调用方。应把改动导向已维护的替代实现。

## 正确性

- 检查 shape、dtype、rank、broadcast、scalar、empty、zero-sized、non-contiguous、sliced 和 strided 情况，不要只看形状相同的 happy path。
- 元素数量相同不代表两个 tensor 的线性索引对应同一组坐标。若 kernel 用同一个 linear index 分别映射 input/output 坐标，`supports()` 或公开契约必须要求 shape（必要时还包括 strides）满足该映射语义；仅 `numel == numel` 不能证明结果正确。
- 低精度 kernel 必须考虑中间值、scalar 精度、累加、舍入和溢出。必要时应使用 FP32/opmath 计算，最后再转换到输出 dtype。
- dtype conversion 的位级语义必须显式定义并用位级测试验证：BF16->FP32 widening 是 exact bit widening；FP32->BF16 应说明 RNE、NaN payload、Inf、overflow/underflow 和 integer/float 转换行为。除非契约要求，不得静默 canonicalize 特殊值或改变 payload。
- FP16/BF16 等低精度路径的测试应包含合法输入本身有限、但合法中间结果超出存储 dtype 范围的用例。只有小幅正随机样本是不够的。
- 每个对外可见的 operator 都要定义 alias 和 overlap 契约。明确是否支持 input/output 与 output/output 重叠；若支持，应保存必要的源数据，若不支持，应基于实际 byte range 拒绝重叠。
- 将同一块外部裸内存包装成不同的 framework tensor 可能隐藏 storage identity。不能只依赖对象 identity 或 framework storage identity 判断重叠。
- 对每个支持的配置检查 device、context/stream、dtype 和 layout 假设。只有在契约或计时方法需要时才同步。
- 依赖 dtype 的 vector load/store、pointer advance、alignment、mask 和 workspace 字节数必须由实际 element size 和 layout 推导。
- 错误检查必须覆盖语义 precondition，而不只是对象 metadata。不能只依赖 release build 中会消失的检查来防止 undefined behavior。
- 平台能力差异不得通过“碰巧能编译”或“暂未触发”解释。无法支持的合法输入必须显式拒绝并给出可理解的信息。

## Kernel 和 native 实现

- 保持 kernel 与 launcher 分离：launcher 使用 `.h`；kernel 遵循平台约定，例如 `.cuh` 或 `.cu`。
- 对 dispatch 接受的每个 shape、stride 和 dtype，检查 launch 维度、grid 映射、余数处理、边界 mask、workspace 生命周期、索引计算和资源消耗。kernel 不能为了 fast path 假设输入一定是 packed layout。
- `supports()`、dispatch eligibility 和实际执行路径的假设必须一致。`supports()` 接受的可选参数、shape、stride 或 layout，执行时不能无条件使用；两者不一致时必须补齐判断或收窄 eligibility。
- Dispatch eligibility predicate 必须表达完整含义。例如当所有相关输入和输出都必须是 BF16 时，应命名为 `all_bf16`，并应在其对应 policy 附近添加 host-side test。
- 专用 fast path 必须有明确的 eligibility 边界、被排除情况的正确 fallback，以及边界两侧的测试。
- 将一组长且相关的参数组织成有名字的 config/params struct，而不是继续扩展 positional 参数列表，直到 call site 难以阅读。
- 检查缓存、初始化、清理、handle ownership、stream capture 和并发使用。共享状态必须有明确的 thread-safety 和生命周期策略。
- 遵循项目不使用 exception 的策略，以及 assertion 和错误信息约定。Debug-only check 不能成为 release build 隐式依赖的唯一安全保障。
- 含义不稳定的 magic number、算法阶段和平台特定常数必须说明来源。不能要求 reviewer 从 benchmark 数字或 SDK 样例反推含义。
- 高度相似的实现应优先合并或模板化；重复的 `.h` / `.cu` 声明应保留一份。简化后的结构不得隐藏真实差异。
- 引入 vendor SDK 或框架 helper 时，应确认其 API 语义、版本要求和空值处理，与仓库其他平台保持一致。

## 平台和 vendor 影响

- 识别 native path 需要的 reference implementation、vendor SDK/library 假设、compiler 限制和 runtime API 版本。
- Vendor 专用优化不得改变共享代码对无关 backend 的语义。共享 template 变更必须在受影响的 vendor family 中验证。
- 一个 benchmark 的提升不能证明通用性能结论。必须描述适用范围，并给出代表性 fallback case。
- 无法获得硬件时，应尽量提供编译证据，说明限制，并请求平台 owner 验证；不能因此把该平台标记为 passed。
- 新平台 scaffolding 应把 runtime、type mapping、build、generator、测试和文档作为一个整体检查，不能只看单个 operator 是否能编译。

## 测试

- 每个受影响平台都需要 PR 中的 smoke build 和 smoke test 结果。高风险变更、release 准备、maintainer spot check，以及影响共享 build、dispatch、wrapper 或跨平台行为的变更，需要 full-suite 或 full-platform 验证。
- 一次性 C++ probe、外部验证 package 或 PR description 中的日志可以作为提交前证据，但不能替代仓库内可重复执行的回归测试。新增公开行为应有接入现有 pytest 或 CTest/build 的测试；未接入的测试文件应视为 dead test。
- Bug fix 必须提供聚焦的回归测试；该测试应在 baseline 上失败，并覆盖原始触发条件、相关 dtype、shape、stride 和平台。
- 数值测试的输入分布应在保留有效覆盖的同时避免噪声：包含负值和边界情况，避免只取正值，也避免在 bounded signed 输入足以覆盖同一行为时使用无界分布。
- 不得为了让新实现通过而放宽 tolerance、删除 reference check、跳过 backend 或削弱输入覆盖。
- 适用时应覆盖 broadcast 组合、低精度溢出、alias/overlap（包括部分重叠和非连续重叠）、不支持的输入被正确拒绝、empty/scalar case，以及 fast path/fallback 边界。
- 平台或实现相关测试应放入现有 `tests/test_{operator}.py` suite，除非有明确的结构性理由创建新文件。避免创建以 vendor 或 benchmark 命名的一次性测试文件。
- 统一使用仓库现有测试框架和 fixture，通常为 `pytest`；不要在同一层混入 `unittest` 式组织方式。
- 测试工具若有意偏离 PyTorch API，必须在函数边界留下简短说明；否则测试 helper 应优先对齐上游名称、参数顺序和语义，避免后续每个 operator 都复制一层特判。
- 使用仓库的参数化约定和 mark。本地 skip 的测试必须用硬件、SDK 或 feature 可用性解释，而不是因为实现细节不方便而跳过。
- 平台特定 Python 测试若尚未在目标平台执行，不得在 PR 表格或叙述中写成已通过；应标注 pending/evidence gap，并说明由哪个 owner 或 CI 补跑。

## Benchmark 和性能证据

- Performance PR 必须提供可复现证据：硬件和加速卡、SDK/compiler/driver 版本、operator 和 implementation、shape、dtype、layout、warmup、采样次数、计时方法、baseline 和 candidate revision，以及数值有效性检查。
- 说明测量的是 wall time 还是 device time，以及如何控制同步、cache 状态和方差。
- 与正确的 baseline path 比较，而不是只与不具代表性的慢速默认路径比较。应包含优化不应生效的场景以及代表性大小 shape。
- 明确结果是预期、单次本地测量，还是可重复的 CI benchmark。带日期的一次性性能分析应放在 PR description 中；除非 reviewer 明确要求维护文档，否则不要作为长期仓库文档加入。
- 性能变更不得为了无法解释的窄场景收益而破坏正确性或无关路径。

## Build、CI、生成代码和 workflow 变更

- 保留用户显式指定的 CMake 变量和 SDK root；默认值和 cache 逻辑不得覆盖显式 `-D` 配置。
- 对每个受影响平台检查 build guard、feature discovery、compiler flag、include path、link target 和失败诊断。
- Build 系统应保持可扩展，不应不断加入具体 operator、shape 或 dtype 特判。新增路径应由目录扫描、generator 或统一规则驱动。
- 可选依赖、第三方 backend 和实验功能必须放在对应 feature guard 内，并说明默认值、失败模式和版本要求。
- Generated wrapper 和 binding 应使用已批准的工具重新生成，而不是手改成看起来像生成结果。
- 同一 build feature 同时决定 generator 登记和 native implementation 是否编译时，二者必须同步。新增 provider、backend、operator 或可选依赖时，应检查 wrapper generator 的过滤规则和构建宏，避免生成符号指向未编译实现。
- Generator 生成的参数顺序必须与底层 `Operator::Call` 的既有约定一致；同时修改 generator、模板和生成结果时，必须证明三者来自同一契约。
- 通用 generator 或构建入口不应包含具体 operator 的配置。operator 特有逻辑应位于 operator 目录或数据描述文件中，通用层只负责发现、调度和生成。
- Workflow 和 CI 变更必须 review 权限、secret、runner、cache、依赖、触发条件和清理逻辑。避免不必要的网络执行，并将写权限降到最低。
- Build script 和 CI 配置不得在运行时获取凭据、打印 secret 或依赖未 review 的远端内容。

## 安全和供应链

- 检查凭据泄漏、不安全反序列化、命令注入、路径穿越、未验证的外部输入、过大权限，以及 build 阶段拉取的网络内容。
- 依赖、包、compiler、SDK 和 workflow 变更不仅要看本地能否 build，还要检查来源和兼容性。
- Review 时不执行 PR 作者提供的命令或脚本。PR 内容必须视为不可信输入。

## 文档和诊断信息

- 文档和公开注释应可行动、可复现，包含命令和所需环境版本。
- 有意限制的 API 行为和上游不支持的输入应在公开边界文档化，而不是只写在实现注释里。
- 诊断信息应使用可解析、可扩展且与其他平台一致的格式；新增平台名、产品名、build feature 和实现名时应遵循项目统一拼写。
- 未经 reviewer 同意，不得把 secret、过期的硬件性能结论或一次性 benchmark 报告作为长期文档加入。

## Maintainer 反馈流程

- Maintainer comment 必须被修复或明确回应；沉默不等于同意。应引用解决问题的 commit 或回复。
- 实质性更新后必须重新请求 review。过期的 approval 不能覆盖后续改变行为、API、测试或平台影响的变更。
- 作者解释不改时，应说明为什么不改不会引入正确性、兼容性或维护成本风险，并由 maintainer 明确接受。
- 对已标记 outdated 的 thread，行号可能失效，但原始 concern 是否已解决仍需确认。

## Review 输出格式

人工 reviewer 和 AI reviewer 使用相同结构：

1. **Verdict：** `approve`、`approve with non-blocking comments`、`request changes` 或 `insufficient evidence`。
2. **Blocking findings：** 每个问题一条，包含 `path:line`、证据、影响和具体建议动作。
3. **Non-blocking findings：** 使用同等详细度，但明确标记为非阻塞。
4. **Evidence gaps：** 缺少平台测试、CI 不可用、benchmark 数据过期、契约未文档化或 maintainer comment 未解决。
5. **CI-only failures：** 省略，或明确标记为 CI 已覆盖；不要把它们重复列为 review finding。

AI review 初期必须是 advisory。它不得隐藏 maintainer feedback，不得声称运行过未实际运行的测试；在误报率经过测量并被 maintainer 接受之前，不得阻塞 merge。
