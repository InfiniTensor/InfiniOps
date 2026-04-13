# InfiniOps

InfiniOps is a high-performance, cross-platform operator library supporting multiple backends: CPU, Nvidia, MetaX, Iluvatar, Moore, Cambricon, and more.

## Prerequisites

- C++17 compatible compiler
- CMake 3.18+
- Python 3.10+
- Hardware-specific SDKs (e.g. CUDA Toolkit, MUSA Toolkit)

## Installation

Install with pip (recommended):

```bash
pip install .
```

This auto-detects available platforms on supported backends. To specify platforms explicitly:

```bash
pip install . -C cmake.define.WITH_CPU=ON -C cmake.define.WITH_NVIDIA=ON
```

### CMake Options

| Option | Description | Default |
|---|---|:-:|
| `-DWITH_CPU=[ON\|OFF]` | Compile the CPU implementation | OFF |
| `-DWITH_NVIDIA=[ON\|OFF]` | Compile the Nvidia implementation | OFF |
| `-DWITH_METAX=[ON\|OFF]` | Compile the MetaX implementation | OFF |
| `-DWITH_ILUVATAR=[ON\|OFF]` | Compile the Iluvatar implementation | OFF |
| `-DWITH_MOORE=[ON\|OFF]` | Compile the Moore implementation | OFF |
| `-DWITH_CAMBRICON=[ON\|OFF]` | Compile the Cambricon implementation | OFF |
| `-DAUTO_DETECT_DEVICES=[ON\|OFF]` | Auto-detect available platforms | ON |

If no accelerator options are provided and auto-detection finds nothing, `WITH_CPU` is enabled by default.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style, commit conventions, PR workflow, development guide, and troubleshooting.

## License

See [LICENSE](LICENSE).
