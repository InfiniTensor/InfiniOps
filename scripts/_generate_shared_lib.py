from _operator_utils import snake_to_pascal


def generate_shared_lib(operator, paths):
    cpp_name = snake_to_pascal(operator.name)

    def _generate_impl_includes():
        return "\n".join(
            f'#include "{str(path).removeprefix("src/")}"' for path in paths
        )

    def _generate_params(node):
        return ", ".join(
            f"{arg.type.spelling} {arg.spelling}" for arg in node.get_arguments()
        )

    def _generate_arguments(node):
        return ", ".join(arg.spelling for arg in node.get_arguments())

    def _generate_make_decl(constructor):
        params = _generate_params(constructor)
        if params:
            params = f"const Config& config, {params}"
        else:
            params = "const Config& config"
        return f"std::unique_ptr<OperatorBase> Make{cpp_name}({params})"

    def _generate_make_def(constructor):
        args = _generate_arguments(constructor)
        make_args = f"config, {args}" if args else "config"
        return f"""{_generate_make_decl(constructor)} {{
  return Operator<{cpp_name}>::make({make_args});
}}"""

    impl_includes = _generate_impl_includes()

    make_defs = "\n\n".join(_generate_make_def(c) for c in operator.constructors)

    source = f"""#include "base/{operator.name}.h"
{impl_includes}

namespace infini::ops {{

{make_defs}

}}  // namespace infini::ops
"""

    make_decls = "\n\n".join(
        f"{_generate_make_decl(c)};" for c in operator.constructors
    )

    return source, make_decls


def generate_shared_lib_header(all_decls):
    combined = "\n\n".join(all_decls)
    return f"""#ifndef INFINI_OPS_MAKE_H_
#define INFINI_OPS_MAKE_H_

#include <memory>
#include <optional>

#include "config.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {{

{combined}

}}  // namespace infini::ops

#endif
"""
