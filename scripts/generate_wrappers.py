import argparse
import concurrent.futures
import functools
import json
import os
import pathlib
import re
import shutil
import subprocess
import textwrap

try:
    import clang.cindex
    from clang.cindex import CursorKind
except ImportError:
    clang = None
    CursorKind = None

_SRC_DIR = pathlib.Path("src")

_BASE_DIR = _SRC_DIR / "base"

_GENERATION_DIR = pathlib.Path("generated")

# Base headers emitted by `generate_torch_ops.py` live alongside the
# hand-written ones in `src/base/`, but in a parallel tree under
# `generated/base/` so they are not committed.
_GENERATED_BASE_DIR = _GENERATION_DIR / "base"

_BINDINGS_DIR = _GENERATION_DIR / "bindings"

_GENERATED_SRC_DIR = _GENERATION_DIR / "src"

_INCLUDE_DIR = _GENERATION_DIR / "include"

_PUBLIC_INCLUDE_DIR = _INCLUDE_DIR / "infini"

_INDENTATION = "  "

_C_API_OPERATOR_NAMES = frozenset({"add"})


@functools.lru_cache(maxsize=1)
def _get_system_include_flags():
    """Probe the system C++ compiler for default include paths so libclang
    can resolve standard headers when parsing an op's base header."""
    compilers = []

    for compiler in ("clang++", "g++"):
        if shutil.which(compiler) is not None:
            compilers.append(compiler)

    system_include_flags = []

    for compiler in compilers:
        for line in subprocess.getoutput(
            f"{compiler} -E -x c++ -v /dev/null"
        ).splitlines():
            if not line.startswith(" "):
                continue

            system_include_flags.append("-isystem")
            system_include_flags.append(line.strip())

    return tuple(system_include_flags)


def _find_base_header(op_name):
    """Resolve the base header for `op_name`, preferring the hand-written
    `src/base/<op>.h` over the auto-generated `generated/base/<op>.h`.
    Mirrors the include-path resolution order used at compile time."""
    src_path = _BASE_DIR / f"{op_name}.h"

    if src_path.exists():
        return src_path

    generated_path = _GENERATED_BASE_DIR / f"{op_name}.h"

    if generated_path.exists():
        return generated_path

    raise FileNotFoundError(f"no base header for op {op_name!r}")


class _ParsedType:
    def __init__(self, spelling):
        self.spelling = spelling


class _ParsedArgument:
    def __init__(self, type_spelling, spelling):
        self.type = _ParsedType(type_spelling)
        self.spelling = spelling


class _ParsedFunction:
    def __init__(self, arguments):
        self._arguments = arguments

    def get_arguments(self):
        return self._arguments


class _OperatorExtractor:
    def __call__(self, op_name):
        if clang is None:
            return _parse_operator_header(op_name)

        index = clang.cindex.Index.create()
        args = (
            "-std=c++17",
            "-x",
            "c++",
            "-I",
            "src",
            "-I",
            str(_GENERATION_DIR),
        ) + _get_system_include_flags()
        translation_unit = index.parse(str(_find_base_header(op_name)), args=args)

        nodes = tuple(type(self)._find(translation_unit.cursor, op_name))

        constructors = []
        calls = []

        for node in nodes:
            if node.kind == CursorKind.CONSTRUCTOR:
                constructors.append(node)
            elif node.kind == CursorKind.CXX_METHOD and node.spelling == "operator()":
                calls.append(node)

        return _Operator(op_name, constructors, calls)

    @staticmethod
    def _find(node, op_name):
        pascal_case_op_name = _snake_to_pascal(op_name)

        if (
            node.semantic_parent
            and node.semantic_parent.spelling == pascal_case_op_name
        ):
            yield node

        for child in node.get_children():
            yield from _OperatorExtractor._find(child, op_name)


def _parse_operator_header(op_name):
    pascal_case_op_name = _snake_to_pascal(op_name)
    source = _strip_cpp_comments(_find_base_header(op_name).read_text())
    class_body = _extract_class_body(source, pascal_case_op_name)
    constructors = [
        _ParsedFunction(_parse_parameter_list(params))
        for params in _find_signature_parameters(
            class_body, rf"(?:explicit\s+)?{pascal_case_op_name}\s*\("
        )
    ]
    calls = [
        _ParsedFunction(_parse_parameter_list(params))
        for params in _find_signature_parameters(
            class_body, r"(?:virtual\s+)?void\s+operator\s*\(\s*\)\s*\("
        )
    ]

    return _Operator(op_name, constructors, calls)


def _strip_cpp_comments(source):
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//.*", "", source)


def _extract_class_body(source, class_name):
    match = re.search(rf"\bclass\s+{class_name}\b[^{{]*{{", source)

    if match is None:
        raise ValueError(f"no class definition for {class_name!r}")

    start = match.end()
    depth = 1
    index = start

    while index < len(source):
        char = source[index]

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index]

        index += 1

    raise ValueError(f"unterminated class definition for {class_name!r}")


def _find_signature_parameters(source, pattern):
    params = []

    for match in re.finditer(pattern, source):
        opening_paren = match.end() - 1

        if opening_paren < 0 or source[opening_paren] != "(":
            continue

        closing_paren = _find_matching_delimiter(source, opening_paren, "(", ")")
        params.append(source[opening_paren + 1 : closing_paren])

    return params


def _find_matching_delimiter(source, start, opening, closing):
    depth = 0

    for index in range(start, len(source)):
        char = source[index]

        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index

    raise ValueError(f"unmatched delimiter {opening!r}")


def _parse_parameter_list(params):
    arguments = []

    for param in _split_top_level(params, ","):
        param = _strip_default_argument(param.strip())

        if not param or param == "void":
            continue

        match = re.match(r"(.+?[\s*&]+)([A-Za-z_][A-Za-z0-9_]*)$", param)

        if match is None:
            raise ValueError(f"could not parse parameter {param!r}")

        arguments.append(_ParsedArgument(match.group(1).strip(), match.group(2)))

    return arguments


def _split_top_level(text, delimiter):
    parts = []
    start = 0
    depth = 0
    pairs = {"<": ">", "(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in pairs.items()}

    for index, char in enumerate(text):
        if char in pairs:
            depth += 1
        elif char in closing:
            depth -= 1
        elif char == delimiter and depth == 0:
            parts.append(text[start:index])
            start = index + 1

    parts.append(text[start:])
    return parts


def _strip_default_argument(param):
    parts = _split_top_level(param, "=")
    return parts[0].strip()


class _Operator:
    def __init__(self, name, constructors, calls):
        self.name = name

        self.constructors = constructors

        self.calls = calls


def _find_optional_tensor_params(op_name):
    """Return a set of parameter names declared as `std::optional<Tensor>` in
    the base header. `libclang` resolves the type to `int` when the STL
    headers are not fully available, so we fall back to a regex scan of the
    source text.
    """
    source = _find_base_header(op_name).read_text()

    return set(re.findall(r"std::optional<Tensor>\s+(\w+)", source))


def _find_vector_tensor_params(op_name):
    """Return a set of parameter names declared as `std::vector<Tensor>` in
    the base header.
    """
    source = _find_base_header(op_name).read_text()

    return set(re.findall(r"std::vector<Tensor>\s+(\w+)", source))


def _find_vector_int64_params(op_name):
    """Return a set of parameter names declared as `std::vector<int64_t>` in
    the base header.

    libclang on systems where the STL headers are not fully indexable
    silently falls back to reporting the type as `int` for these params,
    which then leaks into the generated bindings as `const int padding`
    instead of `const std::vector<int64_t> padding` and breaks the call
    to the base operator.  Regex-scan the source so the binding's
    parameter type comes from the actual declaration.
    """
    source = _find_base_header(op_name).read_text()

    return set(re.findall(r"std::vector<int64_t>\s+(\w+)", source))


def _generate_pybind11(operator):
    optional_tensor_params = _find_optional_tensor_params(operator.name)
    vector_tensor_params = _find_vector_tensor_params(operator.name)
    vector_int64_params = _find_vector_int64_params(operator.name)

    def _is_optional_tensor(arg):
        if arg.spelling in optional_tensor_params:
            return True

        return "std::optional" in arg.type.spelling and "Tensor" in arg.type.spelling

    def _is_optional(arg):
        return "std::optional" in arg.type.spelling

    def _is_vector_tensor(arg):
        if arg.spelling in vector_tensor_params:
            return True

        return "std::vector" in arg.type.spelling and "Tensor" in arg.type.spelling

    def _is_vector_int64(arg):
        return arg.spelling in vector_int64_params

    def _generate_params(node):
        parts = []

        for arg in node.get_arguments():
            if arg.spelling == "stream":
                continue

            if _is_optional_tensor(arg):
                parts.append(f"std::optional<py::object> {arg.spelling}")
            elif _is_vector_tensor(arg):
                parts.append(f"std::vector<py::object> {arg.spelling}")
            elif _is_vector_int64(arg):
                parts.append(f"const std::vector<int64_t> {arg.spelling}")
            else:
                param = arg.type.spelling.replace("const Tensor", "py::object").replace(
                    "Tensor", "py::object"
                )
                parts.append(f"{param} {arg.spelling}")

        return ", ".join(parts)

    def _generate_arguments(node):
        args = []

        for arg in node.get_arguments():
            if arg.spelling == "stream":
                continue

            if _is_optional_tensor(arg):
                args.append(f"OptionalTensorFromPybind11Handle({arg.spelling})")
            elif _is_vector_tensor(arg):
                args.append(f"VectorTensorFromPybind11Handle({arg.spelling})")
            elif "Tensor" in arg.type.spelling:
                args.append(f"TensorFromPybind11Handle({arg.spelling})")
            else:
                args.append(arg.spelling)

        return ", ".join(args)

    op_name = operator.name
    pascal_case_op_name = _snake_to_pascal(op_name)

    def _generate_init(constructor):
        constructor_params = _generate_params(constructor)

        return f"""      .def(py::init([]({constructor_params}) {{
        Config config;
        return std::unique_ptr<Self>{{static_cast<Self*>(generated_dispatch::Make{pascal_case_op_name}(config, {_generate_arguments(constructor)}).release())}};
      }}))"""

    def _generate_py_args(node):
        parts = []

        for arg in node.get_arguments():
            if arg.spelling == "stream":
                continue

            if _is_optional(arg):
                parts.append(f'py::arg("{arg.spelling}") = py::none()')
            else:
                parts.append(f'py::arg("{arg.spelling}")')

        return ", ".join(parts)

    def _generate_call(op_name, call, method=True):
        call_params = _generate_params(call)
        call_args = _generate_arguments(call)

        if not method:
            params = (
                f"{call_params}, std::uintptr_t stream, std::size_t implementation_index"
                if call_params
                else "std::uintptr_t stream, std::size_t implementation_index"
            )
            py_args = _generate_py_args(call)
            py_args_str = f"{py_args}, " if py_args else ""

            return (
                f'  m.def("{op_name}", []({params}) {{\n'
                f"    Handle handle;\n"
                f"    if (stream) {{\n"
                f"      handle.set_stream(reinterpret_cast<void*>(stream));\n"
                f"    }}\n"
                f"    Config config;\n"
                f"    config.set_implementation_index(implementation_index);\n"
                f"    return functional::{pascal_case_op_name}(handle, config, {call_args});\n"
                f'  }}, {py_args_str}py::kw_only(), py::arg("stream") = 0, py::arg("implementation_index") = 0);'
            )

        # The first lambda parameter is conventionally named `self`, but
        # ATen schemas often have a parameter literally called `self`
        # (e.g. `pow.Tensor_Scalar_out(Scalar self, Tensor exponent)`),
        # so rename to `op` to avoid the collision in the generated code.

        return f"""      .def("__call__", [](const Self& op, {call_params}) {{
        return generated_dispatch::Invoke{pascal_case_op_name}(op, {call_args});
      }})"""

    def _overload_order_key(node):
        """Sort key that places more-specific overloads first.

        Tensor parameters are exposed to pybind as `py::object`, which
        accepts any Python value and only fails inside
        `TensorFromPybind11Handle`.  When a class has both Tensor and
        scalar overloads, pybind's overload-resolver tries them in
        registration order and stops at the first that does not raise,
        so the scalar overload must be registered first; otherwise the
        permissive Tensor signature swallows scalar calls and aborts at
        runtime.
        """
        object_like = 0
        total = 0

        for arg in node.get_arguments():
            if arg.spelling == "stream":
                continue

            total += 1

            if (
                _is_optional_tensor(arg)
                or _is_vector_tensor(arg)
                or "Tensor" in arg.type.spelling
            ):
                object_like += 1

        return (object_like, -total)

    constructors = sorted(operator.constructors, key=_overload_order_key)
    operator_calls = sorted(operator.calls, key=_overload_order_key)

    inits = "\n".join(_generate_init(constructor) for constructor in constructors)
    calls = "\n".join(_generate_call(operator.name, call) for call in operator_calls)
    callers = "\n".join(
        _generate_call(operator.name, call, method=False) for call in operator_calls
    )

    return f"""#ifndef INFINI_OPS_BINDINGS_{op_name.upper()}_H_
#define INFINI_OPS_BINDINGS_{op_name.upper()}_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "base/{op_name}.h"
#include "config.h"
#include "infini/ops.h"
#include "generated/bindings/generated_dispatch.h"
#include "handle.h"
#include "pybind11_utils.h"

namespace py = pybind11;

namespace infini::ops {{

void Bind{pascal_case_op_name}(py::module& m) {{
  using Self = {pascal_case_op_name};

  py::class_<Self>(m, "{pascal_case_op_name}")
{inits}
{calls}
      .def_static("active_implementation_indices", [](const std::string& device) {{
        auto dev_type = TryDeviceTypeFromString<Self>(device);
        if (!dev_type.has_value()) {{
          return std::vector<std::size_t>{{}};
        }}
        return generated_dispatch::ActiveImplementationIndicesFor{pascal_case_op_name}(*dev_type);
      }})
      .def_static("clear_cache", &generated_dispatch::ClearCacheFor{pascal_case_op_name});

{callers}
}}

}}  // namespace infini::ops

#endif
"""


def _generate_legacy_c(operator, paths):
    def _generate_source(operator):
        impl_includes = "\n".join(
            f'#include "{_to_include_path(path)}"' for path in paths
        )

        return f"""#include "../../handle.h"
#include "../../tensor.h"
#include "infiniop/ops/{operator.name.lower()}.h"
{impl_includes}

static infini::ops::DataType DataTypeFromInfiniDType(
    const infiniDtype_t& dtype) {{
  static constexpr infini::ops::ConstexprMap<infiniDtype_t,
                                             infini::ops::DataType, 12>
      kInfiniDTypeToDataType{{
          {{{{{{INFINI_DTYPE_I8, infini::ops::DataType::kInt8}},
            {{INFINI_DTYPE_I16, infini::ops::DataType::kInt16}},
            {{INFINI_DTYPE_I32, infini::ops::DataType::kInt32}},
            {{INFINI_DTYPE_I64, infini::ops::DataType::kInt64}},
            {{INFINI_DTYPE_U8, infini::ops::DataType::kUInt8}},
            {{INFINI_DTYPE_U16, infini::ops::DataType::kUInt16}},
            {{INFINI_DTYPE_U32, infini::ops::DataType::kUInt32}},
            {{INFINI_DTYPE_U64, infini::ops::DataType::kUInt64}},
            {{INFINI_DTYPE_F16, infini::ops::DataType::kFloat16}},
            {{INFINI_DTYPE_BF16, infini::ops::DataType::kBFloat16}},
            {{INFINI_DTYPE_F32, infini::ops::DataType::kFloat32}},
            {{INFINI_DTYPE_F64, infini::ops::DataType::kFloat64}}}}}}}};

  return kInfiniDTypeToDataType.at(dtype);
}}

static infini::ops::Device::Type DeviceTypeFromInfiniDevice(
    const infiniDevice_t& device) {{
  static constexpr infini::ops::ConstexprMap<
      infiniDevice_t, infini::ops::Device::Type,
      static_cast<std::size_t>(INFINI_DEVICE_TYPE_COUNT)>
      kInfiniDeviceToDeviceType{{
          {{{{{{INFINI_DEVICE_CPU, infini::ops::Device::Type::kCpu}},
            {{INFINI_DEVICE_NVIDIA, infini::ops::Device::Type::kNvidia}},
            {{INFINI_DEVICE_CAMBRICON, infini::ops::Device::Type::kCambricon}},
            {{INFINI_DEVICE_ASCEND, infini::ops::Device::Type::kAscend}},
            {{INFINI_DEVICE_METAX, infini::ops::Device::Type::kMetax}},
            {{INFINI_DEVICE_MOORE, infini::ops::Device::Type::kMoore}},
            {{INFINI_DEVICE_ILUVATAR, infini::ops::Device::Type::kIluvatar}},
            {{INFINI_DEVICE_KUNLUN, infini::ops::Device::Type::kKunlun}},
            {{INFINI_DEVICE_HYGON, infini::ops::Device::Type::kHygon}},
            {{INFINI_DEVICE_QY, infini::ops::Device::Type::kQy}}}}}}}};

  return kInfiniDeviceToDeviceType.at(device);
}}

__C {_generate_create_func_def(operator)}

__C {_generate_get_workspace_size_func_def(operator)}

__C {_generate_call_func_def(operator)}

__C {_generate_destroy_func_def(operator)}
"""

    def _generate_header(operator):
        return f"""#ifndef __INFINIOP_{operator.name.upper()}_API_H__
#define __INFINIOP_{operator.name.upper()}_API_H__

#include "base/{operator.name.lower()}.h"

typedef struct infini::ops::Operator<infini::ops::{operator.name}> *infiniop{operator.name}Descriptor_t;

__C __export {_generate_create_func_decl(operator)};

__C __export {_generate_get_workspace_size_func_decl(operator)};

__C __export {_generate_call_func_decl(operator)};

__C __export {_generate_destroy_func_decl(operator)};

#endif
"""

    def _generate_create_func_def(operator):
        name = operator.name
        constructor = operator.constructors[-1]

        return f"""{_generate_create_func_decl(operator)} {{
    *desc_ptr = infini::ops::Operator<infini::ops::{name}>::Make({_generate_arguments(constructor)}).release();

    return INFINI_STATUS_SUCCESS;
}}"""

    def _generate_get_workspace_size_func_def(operator):
        return f"""{_generate_get_workspace_size_func_decl(operator)} {{
    *size = 0;  // desc->workspace_size();

    return INFINI_STATUS_SUCCESS;
}}"""

    def _generate_call_func_def(operator):
        call = operator.calls[-1]

        return f"""{_generate_call_func_decl(operator)} {{
    (*desc)(stream, {_generate_arguments(call, is_data=True)});

    return INFINI_STATUS_SUCCESS;
}}"""

    def _generate_destroy_func_def(operator):
        return f"""{_generate_destroy_func_decl(operator)} {{
    delete desc;

    return INFINI_STATUS_SUCCESS;
}}"""

    def _generate_create_func_decl(operator):
        name = operator.name
        constructor = operator.constructors[-1]
        params = _generate_params(constructor)

        return f"infiniStatus_t infiniopCreate{name}Descriptor(infiniopHandle_t handle, infiniop{name}Descriptor_t *desc_ptr, {params})"

    def _generate_get_workspace_size_func_decl(operator):
        name = operator.name

        return f"infiniStatus_t infiniopGet{name}WorkspaceSize(infiniop{name}Descriptor_t desc, size_t *size)"

    def _generate_call_func_decl(operator):
        name = operator.name
        call = operator.calls[-1]
        params = _generate_params(call, call=True)
        params = params.replace("void * stream, ", "")

        return f"infiniStatus_t infiniop{name}(infiniop{name}Descriptor_t desc, void *workspace, size_t workspace_size, {params}, void *stream)"

    def _generate_destroy_func_decl(operator):
        name = operator.name

        return f"infiniStatus_t infiniopDestroy{name}Descriptor(infiniop{name}Descriptor_t desc)"

    def _generate_params(node, call=False):
        arguments = tuple(node.get_arguments())

        arguments = (arguments[-1], *arguments[:-1])

        def _handle_tensor(spelling):
            if call:
                return spelling.replace("Tensor", "void *")

            return spelling.replace("Tensor", "infiniopTensorDescriptor_t")

        def _handle_std_optional(spelling):
            return spelling.replace("std::optional<", "").replace(">", "")

        return ", ".join(
            f"{_handle_std_optional(_handle_tensor(arg.type.spelling))} {arg.spelling}"
            for arg in arguments
        )

    def _generate_arguments(node, is_data=False):
        return ", ".join(
            _generate_tensor_caster(arg.spelling, is_data=is_data)
            if "Tensor" in arg.type.spelling
            else arg.spelling
            for arg in node.get_arguments()
            if arg.spelling != "handle" and arg.spelling != "stream"
        )

    def _generate_tensor_caster(name, is_data=False):
        if is_data:
            return f"infini::ops::Tensor(const_cast<void *>({name}), infini::ops::Tensor::Shape{{}})"

        return f"infini::ops::Tensor{{nullptr, {name}->shape(), DataTypeFromInfiniDType({name}->dtype()), infini::ops::Device{{DeviceTypeFromInfiniDevice(handle->device), handle->device_id}}, {name}->strides()}}"

    return _generate_source(operator), _generate_header(operator)


def _generate_generated_dispatch_entries(operator):
    optional_tensor_params = _find_optional_tensor_params(operator.name)
    vector_tensor_params = _find_vector_tensor_params(operator.name)
    vector_int64_params = _find_vector_int64_params(operator.name)

    def _is_optional_tensor(arg):
        if arg.spelling in optional_tensor_params:
            return True

        return "std::optional" in arg.type.spelling and "Tensor" in arg.type.spelling

    def _is_vector_tensor(arg):
        if arg.spelling in vector_tensor_params:
            return True

        return "std::vector" in arg.type.spelling and "Tensor" in arg.type.spelling

    def _is_vector_int64(arg):
        return arg.spelling in vector_int64_params

    def _generate_params(node):
        parts = []

        for arg in node.get_arguments():
            if arg.spelling == "stream":
                continue

            if _is_optional_tensor(arg):
                parts.append(f"std::optional<Tensor> {arg.spelling}")
            elif _is_vector_tensor(arg):
                parts.append(f"std::vector<Tensor> {arg.spelling}")
            elif _is_vector_int64(arg):
                parts.append(f"std::vector<int64_t> {arg.spelling}")
            else:
                parts.append(f"{arg.type.spelling} {arg.spelling}")

        return ", ".join(parts)

    def _generate_arguments(node):
        return ", ".join(
            arg.spelling for arg in node.get_arguments() if arg.spelling != "stream"
        )

    def _append_optional_args(prefix, args):
        if args:
            return f"{prefix}, {args}"

        return prefix

    def _append_optional_params(prefix, params):
        if params:
            return f"{prefix}, {params}"

        return prefix

    pascal_case_op_name = _snake_to_pascal(operator.name)
    declarations = [
        f"std::vector<std::size_t> ActiveImplementationIndicesFor"
        f"{pascal_case_op_name}(Device::Type dev_type);"
    ]
    definitions = [
        f"""std::vector<std::size_t> ActiveImplementationIndicesFor{pascal_case_op_name}(Device::Type dev_type) {{
  return Operator<{pascal_case_op_name}>::active_implementation_indices(dev_type);
}}"""
    ]

    declarations.append(f"void ClearCacheFor{pascal_case_op_name}();")
    definitions.append(
        f"""void ClearCacheFor{pascal_case_op_name}() {{
  Operator<{pascal_case_op_name}>::clear_cache();
}}"""
    )

    for constructor in operator.constructors:
        params = _generate_params(constructor)
        args = _generate_arguments(constructor)

        declarations.append(
            f"std::unique_ptr<Operator<{pascal_case_op_name}>> "
            f"Make{pascal_case_op_name}("
            f"{_append_optional_params('const Config& config', params)});"
        )
        definitions.append(
            f"""std::unique_ptr<Operator<{pascal_case_op_name}>> Make{pascal_case_op_name}({_append_optional_params("const Config& config", params)}) {{
  return Operator<{pascal_case_op_name}>::Make({_append_optional_args("config", args)});
}}"""
        )

    for call in operator.calls:
        params = _generate_params(call)
        args = _generate_arguments(call)

        declarations.append(
            f"void Invoke{pascal_case_op_name}(const "
            f"{_append_optional_params(f'{pascal_case_op_name}& op', params)});"
        )
        definitions.append(
            f"""void Invoke{pascal_case_op_name}(const {_append_optional_params(f"{pascal_case_op_name}& op", params)}) {{
  return static_cast<const Operator<{pascal_case_op_name}>&>(op)({args});
}}"""
        )

        declarations.append(
            f"void Call{pascal_case_op_name}(const Handle& handle, "
            f"{_append_optional_params('const Config& config', params)});"
        )
        definitions.append(
            f"""void Call{pascal_case_op_name}(const Handle& handle, {_append_optional_params("const Config& config", params)}) {{
  return Operator<{pascal_case_op_name}>::Call({_append_optional_args("handle, config", args)});
}}"""
        )

    return declarations, definitions


def _generate_functional_entries(operator):
    def _generate_params(node):
        return ", ".join(
            f"{arg.type.spelling} {arg.spelling}"
            for arg in node.get_arguments()
            if arg.spelling != "stream"
        )

    def _generate_arguments(node):
        return ", ".join(
            arg.spelling for arg in node.get_arguments() if arg.spelling != "stream"
        )

    def _append_optional_args(prefix, args):
        if args:
            return f"{prefix}, {args}"

        return prefix

    def _append_optional_params(prefix, params):
        if params:
            return f"{prefix}, {params}"

        return prefix

    pascal_case_op_name = _snake_to_pascal(operator.name)
    op_type = f"::infini::ops::{pascal_case_op_name}"
    operator_type = f"::infini::ops::Operator<{op_type}>"
    declarations = []
    definitions = []

    for call in operator.calls:
        params = _generate_params(call)
        args = _generate_arguments(call)
        function_params = _append_optional_params(
            "const Handle& handle, const Config& config", params
        )

        declarations.append(f"void {pascal_case_op_name}({function_params});")
        definitions.append(
            f"""void {pascal_case_op_name}({function_params}) {{
  return {operator_type}::Call({_append_optional_args("handle, config", args)});
}}"""
        )

    return declarations, definitions


def _generate_c_api_entries(operator):
    pascal_case_op_name = _snake_to_pascal(operator.name)
    declarations = []
    definitions = []

    if operator.name not in _C_API_OPERATOR_NAMES:
        return declarations, definitions

    for call in operator.calls:
        params = _generate_c_api_params(call)
        validations = _generate_c_api_validations(call)
        args = _generate_c_api_arguments(call)
        signature = _format_c_api_signature(f"infiniOps{pascal_case_op_name}", params)
        declarations.append(f"INFINI_OPS_API {signature};")
        definitions.append(
            f"""INFINI_OPS_API {signature} {{
  try {{
{validations}
    const infini::ops::Handle default_handle;
    const infini::ops::Config default_config;
    infini::ops::functional::{pascal_case_op_name}(
        handle == nullptr ? default_handle : handle->handle,
        config == nullptr ? default_config : config->config{args});
    SetLastError("");
    return INFINI_OPS_STATUS_SUCCESS;
  }} catch (const std::bad_alloc&) {{
    SetLastError("out of memory while running `infiniOps{pascal_case_op_name}`");
    return INFINI_OPS_STATUS_OUT_OF_MEMORY;
  }} catch (const std::exception& error) {{
    SetLastError(error.what());
    return INFINI_OPS_STATUS_INTERNAL_ERROR;
  }} catch (...) {{
    SetLastError("unknown error while running `infiniOps{pascal_case_op_name}`");
    return INFINI_OPS_STATUS_INTERNAL_ERROR;
  }}
}}"""
        )

    return declarations, definitions


def _generate_c_api_params(node):
    params = ["InfiniOpsHandle handle", "InfiniOpsConfig config"]

    for arg in node.get_arguments():
        if arg.spelling == "stream":
            continue

        params.append(_c_api_param(arg))

    return params


def _format_c_api_signature(name, params):
    return (
        f"InfiniOpsStatus {name}(\n    {', '.join(params[:2])},\n    "
        + ",\n    ".join(params[2:])
        + ")"
    )


def _c_api_param(arg):
    if arg.type.spelling == "const Tensor":
        return f"const InfiniOpsTensor* {arg.spelling}"

    if arg.type.spelling == "Tensor":
        return f"InfiniOpsTensor* {arg.spelling}"

    raise ValueError(
        f"unsupported C API parameter {arg.spelling!r}: {arg.type.spelling!r}"
    )


def _generate_c_api_validations(node):
    lines = []

    for arg in node.get_arguments():
        if arg.spelling == "stream":
            continue

        if arg.type.spelling not in {"const Tensor", "Tensor"}:
            continue

        lines.extend(
            (
                f"    InfiniOpsStatus {arg.spelling}_status = "
                f'ValidateTensor("{arg.spelling}", {arg.spelling});',
                f"    if ({arg.spelling}_status != INFINI_OPS_STATUS_SUCCESS) {{",
                f"      return {arg.spelling}_status;",
                "    }",
            )
        )

    return "\n".join(lines)


def _generate_c_api_arguments(node):
    args = [
        f"ToInternalTensor(*{arg.spelling})"
        for arg in node.get_arguments()
        if arg.spelling != "stream"
    ]

    if not args:
        return ""

    return ",\n        " + ",\n        ".join(args)


def _generate_generated_dispatch_header(op_names, devices, declarations):
    header_base_includes = "\n".join(
        f'#include "base/{op_name}.h"' for op_name in op_names
    )
    header_device_includes = "\n".join(
        f'#include "{path}"' for path in _device_marker_headers(devices)
    )

    return f"""#ifndef INFINI_OPS_GENERATED_BINDINGS_GENERATED_DISPATCH_H_
#define INFINI_OPS_GENERATED_BINDINGS_GENERATED_DISPATCH_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "config.h"
#include "device.h"
#include "handle.h"
#include "operator.h"

{header_device_includes}

{header_base_includes}

namespace infini::ops::generated_dispatch {{

{chr(10).join(declarations)}

}}  // namespace infini::ops::generated_dispatch

#endif
"""


def _generate_generated_dispatch_source(impl_paths, definitions):
    impl_includes = "\n".join(f'#include "{impl_path}"' for impl_path in impl_paths)

    return f"""#include "generated_dispatch.h"

// clang-format off
{impl_includes}
// clang-format on

namespace infini::ops::generated_dispatch {{

{chr(10).join(definitions)}

}}  // namespace infini::ops::generated_dispatch
"""


def _generate_functional_header(declarations):
    return f"""#ifndef INFINI_OPS_FUNCTIONAL_OPS_H_
#define INFINI_OPS_FUNCTIONAL_OPS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "config.h"
#include "data_type.h"
#include "device.h"
#include "handle.h"
#include "tensor.h"

namespace infini::ops::functional {{

{chr(10).join(declarations)}

}}  // namespace infini::ops::functional

#endif
"""


def _generate_functional_source(op_names, devices, impl_paths, definitions):
    base_includes = "\n".join(f'#include "base/{op_name}.h"' for op_name in op_names)
    device_includes = "\n".join(
        f'#include "{path}"' for path in _device_marker_headers(devices)
    )
    impl_includes = "\n".join(
        f'#include "{_to_include_path(impl_path)}"' for impl_path in impl_paths
    )

    return f"""#include "infini/functional_ops.h"

// clang-format off
{device_includes}
{base_includes}
{impl_includes}
// clang-format on

namespace infini::ops::functional {{

{chr(10).join(definitions)}

}}  // namespace infini::ops::functional
"""


def _generate_c_api_header(declarations):
    return f"""#ifndef INFINI_OPS_C_OPS_H_
#define INFINI_OPS_C_OPS_H_

{chr(10).join(declarations)}

#endif
"""


def _generate_c_api_source(definitions):
    return f"""// Generated C ABI operator wrappers.

{chr(10).join(definitions)}
"""


def _device_marker_headers(devices):
    paths = {
        "cpu": "native/cpu/device_.h",
        "nvidia": "native/cuda/nvidia/device_.h",
        "cambricon": "native/cambricon/device_.h",
        "ascend": "native/ascend/device_.h",
        "metax": "native/cuda/metax/device_.h",
        "moore": "native/cuda/moore/device_.h",
        "iluvatar": "native/cuda/iluvatar/device_.h",
    }

    return [paths[device] for device in devices if device in paths]


def _generate_binding_source(op_name):
    return f"""#include "{op_name}.h"
"""


def _snake_to_pascal(snake_str):
    return "".join(word.capitalize() for word in snake_str.split("_"))


def _to_include_path(path):
    text = str(path)

    for prefix in ("src/", "generated/"):
        if text.startswith(prefix):
            return text[len(prefix) :]

    return text


def _matches_scan_dir(impl_path, scan_dirs):
    return any(part in scan_dirs for part in impl_path.parts)


_OPERATOR_DECL_RE = re.compile(r"\bclass\s+Operator<\s*([A-Za-z_][A-Za-z0-9_]*)\b")


def _index_impl_headers(impl_roots, scan_dirs):
    """Index implementation headers by base operator class name.

    The previous implementation scanned every implementation header once per
    operator.  With the generated PyTorch backend enabled this becomes hundreds
    of ops times hundreds of headers during CMake configure.  Read each header
    once instead and keep the same insertion order as the old nested loops.
    """
    by_operator = {}

    for impl_root in impl_roots:
        for impl_path in impl_root.rglob("*.h"):
            if not _matches_scan_dir(impl_path, scan_dirs):
                continue

            text = impl_path.read_text()

            for match in _OPERATOR_DECL_RE.finditer(text):
                by_operator.setdefault(match.group(1), []).append(impl_path)

    return by_operator


def _get_all_ops(devices, with_torch=False):
    scan_dirs = set(devices)

    if with_torch:
        scan_dirs.add("torch")

    ops = {}

    base_dirs = [_BASE_DIR]

    # Only pull in the auto-generated torch op bases when the build is
    # actually compiling them (`--with-torch`).  Otherwise a stale
    # `generated/` left over from a previous configure (or rsynced into
    # a CI container) would cause `ops.cc` to include base headers for
    # ops that have no compiled implementation, breaking the build.
    if with_torch and _GENERATED_BASE_DIR.exists():
        base_dirs.append(_GENERATED_BASE_DIR)

    impl_roots = [_SRC_DIR]

    if with_torch and (_GENERATION_DIR / "torch").exists():
        impl_roots.append(_GENERATION_DIR)

    impl_headers_by_operator = _index_impl_headers(impl_roots, scan_dirs)

    for base_dir in base_dirs:
        for file_path in base_dir.iterdir():
            if not file_path.is_file():
                continue

            op_name = file_path.stem

            # Hand-written `src/base/` is scanned first; the generated
            # tree never overrides an already-known op.
            if op_name in ops:
                continue

            ops[op_name] = []
            ops[op_name].extend(
                impl_headers_by_operator.get(_snake_to_pascal(op_name), ())
            )

    return ops


def _generate_op_artifacts(item):
    op_name, impl_paths = item
    extractor = _OperatorExtractor()
    operator = extractor(op_name)
    header_name = f"{op_name}.h"
    legacy_c_source, legacy_c_header = _generate_legacy_c(operator, impl_paths)
    dispatch_declarations, dispatch_definitions = _generate_generated_dispatch_entries(
        operator
    )
    functional_declarations, functional_definitions = _generate_functional_entries(
        operator
    )
    c_api_declarations, c_api_definitions = _generate_c_api_entries(operator)

    return {
        "op_name": op_name,
        "header_name": header_name,
        "bind_func_name": f"Bind{_snake_to_pascal(op_name)}",
        "pybind11": _generate_pybind11(operator),
        "binding_source": _generate_binding_source(op_name),
        "legacy_c_source": legacy_c_source,
        "legacy_c_header": legacy_c_header,
        "dispatch_declarations": dispatch_declarations,
        "dispatch_definitions": dispatch_definitions,
        "functional_declarations": functional_declarations,
        "functional_definitions": functional_definitions,
        "c_api_declarations": c_api_declarations,
        "c_api_definitions": c_api_definitions,
        "impl_paths": impl_paths,
    }


def _wrapper_gen_jobs(with_torch):
    raw = os.environ.get("INFINIOPS_WRAPPER_GEN_JOBS")

    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 1

    if not with_torch:
        return 1

    return min(os.cpu_count() or 1, 8)


def _use_monolithic_bindings():
    value = os.environ.get("INFINIOPS_MONOLITHIC_BINDINGS", "")

    return value.upper() in {"1", "ON", "TRUE"}


def _dispatch_gen_batch_size():
    raw = os.environ.get("INFINIOPS_DISPATCH_BATCH_SIZE")

    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 8

    return 8


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An automatic wrapper generator.")

    parser.add_argument(
        "--devices",
        nargs="+",
        default="cpu",
        type=str,
        help="Devices to use. Please pick from `cpu`, `nvidia`, `cambricon`, `ascend`, `metax`, `moore`, `iluvatar`, `kunlun`, `hygon`, and `qy`. (default: `cpu`)",
    )

    parser.add_argument(
        "--with-torch",
        action="store_true",
        help="Include PyTorch C++ backend implementations.",
    )

    args = parser.parse_args()

    # Wipe previous outputs so files for ops that have since been removed
    # from the active set (e.g. when toggling `--with-torch`) do not linger
    # and get globbed by a later build.
    for directory in (_BINDINGS_DIR, _GENERATED_SRC_DIR, _INCLUDE_DIR):
        if directory.exists():
            shutil.rmtree(directory)

        directory.mkdir(parents=True)

    _PUBLIC_INCLUDE_DIR.mkdir(parents=True, exist_ok=True)

    ops_json = pathlib.Path("ops.json")

    if ops_json.exists():
        ops = json.loads(ops_json.read_text())
    else:
        ops = _get_all_ops(args.devices, with_torch=args.with_torch)

    bind_func_names = []

    jobs = _wrapper_gen_jobs(args.with_torch)

    if jobs == 1:
        artifacts = [_generate_op_artifacts(item) for item in ops.items()]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            artifacts = list(executor.map(_generate_op_artifacts, ops.items()))

    op_names = [artifact["op_name"] for artifact in artifacts]
    dispatch_declarations = [
        declaration
        for artifact in artifacts
        for declaration in artifact["dispatch_declarations"]
    ]
    functional_declarations = [
        declaration
        for artifact in artifacts
        for declaration in artifact["functional_declarations"]
    ]
    c_api_declarations = [
        declaration
        for artifact in artifacts
        for declaration in artifact["c_api_declarations"]
    ]
    c_api_definitions = [
        definition
        for artifact in artifacts
        for definition in artifact["c_api_definitions"]
    ]
    use_monolithic_bindings = _use_monolithic_bindings()
    op_includes = []

    for artifact in artifacts:
        op_name = artifact["op_name"]
        source_path = _GENERATED_SRC_DIR / op_name
        header_name = artifact["header_name"]
        bind_func_name = artifact["bind_func_name"]

        (_BINDINGS_DIR / header_name).write_text(artifact["pybind11"])

        if use_monolithic_bindings:
            op_includes.append(f'#include "{header_name}"')
        else:
            (_BINDINGS_DIR / f"{op_name}.cc").write_text(artifact["binding_source"])

        source_path.mkdir(exist_ok=True)
        (_GENERATED_SRC_DIR / op_name / "operator.cc").write_text(
            artifact["legacy_c_source"]
        )
        (_INCLUDE_DIR / header_name).write_text(artifact["legacy_c_header"])

        bind_func_names.append(bind_func_name)

    dispatch_header = _generate_generated_dispatch_header(
        op_names, args.devices, dispatch_declarations
    )
    (_BINDINGS_DIR / "generated_dispatch.h").write_text(dispatch_header)

    functional_header = _generate_functional_header(functional_declarations)
    (_PUBLIC_INCLUDE_DIR / "functional_ops.h").write_text(functional_header)
    c_api_header = _generate_c_api_header(c_api_declarations)
    (_PUBLIC_INCLUDE_DIR / "c_ops.h").write_text(c_api_header)
    c_api_source = _generate_c_api_source(c_api_definitions)
    (_GENERATED_SRC_DIR / "c_ops.inc").write_text(c_api_source)

    dispatch_batch_size = _dispatch_gen_batch_size()

    for dispatch_batch_index, start in enumerate(
        range(0, len(artifacts), dispatch_batch_size)
    ):
        batch = artifacts[start : start + dispatch_batch_size]
        impl_paths = list(
            dict.fromkeys(
                impl_path for artifact in batch for impl_path in artifact["impl_paths"]
            )
        )
        definitions = [
            definition
            for artifact in batch
            for definition in artifact["dispatch_definitions"]
        ]
        dispatch_source = _generate_generated_dispatch_source(impl_paths, definitions)
        (_BINDINGS_DIR / f"generated_dispatch_{dispatch_batch_index}.cc").write_text(
            dispatch_source
        )

        functional_definitions = [
            definition
            for artifact in batch
            for definition in artifact["functional_definitions"]
        ]
        functional_source = _generate_functional_source(
            [artifact["op_name"] for artifact in batch],
            args.devices,
            impl_paths,
            functional_definitions,
        )
        (_GENERATED_SRC_DIR / f"functional_ops_{dispatch_batch_index}.cc").write_text(
            functional_source
        )

    bind_func_calls = "\n".join(
        f"{bind_func_name}(m);" for bind_func_name in bind_func_names
    )

    if use_monolithic_bindings:
        op_includes = "\n".join(op_includes)
        ops_source = f"""#include <pybind11/pybind11.h>

// Generated with `INFINIOPS_MONOLITHIC_BINDINGS=1`.
{op_includes}

namespace infini::ops {{

PYBIND11_MODULE(ops, m) {{
{textwrap.indent(bind_func_calls, _INDENTATION)}
}}

}}  // namespace infini::ops
"""
    else:
        bind_func_declarations = "\n".join(
            f"void {bind_func_name}(pybind11::module& m);"
            for bind_func_name in bind_func_names
        )
        ops_source = f"""#include <pybind11/pybind11.h>

namespace infini::ops {{

{bind_func_declarations}

PYBIND11_MODULE(ops, m) {{
{textwrap.indent(bind_func_calls, _INDENTATION)}
}}

}}  // namespace infini::ops
"""

    (_BINDINGS_DIR / "ops.cc").write_text(ops_source)
