import pathlib
import shutil
import subprocess

import clang.cindex
from clang.cindex import CursorKind

_SRC_DIR = pathlib.Path("src")

_BASE_DIR = _SRC_DIR / "base"


def snake_to_pascal(snake_str):
    return "".join(word.capitalize() for word in snake_str.split("_"))


class Operator:
    def __init__(self, name, constructors, calls):
        self.name = name

        self.constructors = constructors

        self.calls = calls


class OperatorExtractor:
    def __call__(self, op_name):
        def _get_system_include_flags():
            def _get_compilers():
                compilers = []

                for compiler in ("clang++", "g++"):
                    if shutil.which(compiler) is not None:
                        compilers.append(compiler)

                return compilers

            system_include_flags = []

            for compiler in _get_compilers():
                for line in subprocess.getoutput(
                    f"{compiler} -E -x c++ -v /dev/null"
                ).splitlines():
                    if not line.startswith(" "):
                        continue

                    system_include_flags.append("-isystem")
                    system_include_flags.append(line.strip())

            return system_include_flags

        system_include_flags = _get_system_include_flags()

        index = clang.cindex.Index.create()
        args = ("-std=c++17", "-x", "c++", "-I", "src") + tuple(system_include_flags)
        translation_unit = index.parse(f"src/base/{op_name}.h", args=args)

        nodes = tuple(type(self)._find(translation_unit.cursor, op_name))

        constructors = []
        calls = []

        for node in nodes:
            if node.kind == CursorKind.CONSTRUCTOR:
                constructors.append(node)
            elif node.kind == CursorKind.CXX_METHOD and node.spelling == "operator()":
                calls.append(node)

        return Operator(op_name, constructors, calls)

    @staticmethod
    def _find(node, op_name):
        pascal_case_op_name = snake_to_pascal(op_name)

        if (
            node.semantic_parent
            and node.semantic_parent.spelling == pascal_case_op_name
        ):
            yield node

        for child in node.get_children():
            yield from OperatorExtractor._find(child, op_name)


def get_all_ops(devices):
    ops = {}

    for file_path in _BASE_DIR.iterdir():
        if not file_path.is_file():
            continue

        op_name = file_path.stem

        ops[op_name] = []

        for file_path in _SRC_DIR.rglob("*"):
            if not file_path.is_file() or file_path.parent.parent.name not in devices:
                continue

            if f"class Operator<{snake_to_pascal(op_name)}" in file_path.read_text():
                ops[op_name].append(file_path)

    return ops
