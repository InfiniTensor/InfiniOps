<!--
Thanks for contributing to InfiniOps! Please read `CONTRIBUTING.md` before
opening a pull request and fill out every section below. Delete any section
that is genuinely not applicable (and note why), but do not delete the
"Checklist" section — it must be filled in for every PR.

The PR title MUST follow Conventional Commits, e.g.
  feat: add fused RMSNorm kernel
  fix: correct stride handling for batched matmul
See: https://www.conventionalcommits.org/
-->

## Summary

<!--
A concise description of **what** this PR changes. Prefer bullet points over
prose. Reference files with backtick-fenced paths (e.g. `src/cuda/gemm/blas.h`).
-->

-
-

## Motivation

<!--
Explain **why** this change is needed. Link to any related issue, bug, or
discussion. If this is a performance change, include before/after numbers
(hardware, shape, dtype, and the measurement methodology).
-->

Closes #

## Type of Change

<!-- Tick one or more. The type MUST match the Conventional Commits prefix
in the PR title and the branch name (see `CONTRIBUTING.md` §Branches). -->

- [ ] `feat` — new feature / new operator / new platform
- [ ] `fix` — bug fix
- [ ] `perf` — performance improvement (no behavioral change)
- [ ] `refactor` — code restructuring without behavior change
- [ ] `test` — adding or fixing tests only
- [ ] `docs` — documentation only
- [ ] `build` / `ci` — build system or CI configuration
- [ ] `chore` — tooling, formatting, or other non-code changes
- [ ] Breaking change (requires a `!` in the Conventional Commits prefix or a `BREAKING CHANGE:` footer)

## Platforms Affected

<!-- Tick every backend whose code is touched **or** whose behavior may change. -->

- [ ] CPU (`WITH_CPU`)
- [ ] NVIDIA (`WITH_NVIDIA`)
- [ ] Iluvatar (`WITH_ILUVATAR`)
- [ ] MetaX (`WITH_METAX`)
- [ ] Cambricon (`WITH_CAMBRICON`)
- [ ] Moore (`WITH_MOORE`)
- [ ] Ascend (`WITH_ASCEND`)
- [ ] PyTorch C++ bindings (`WITH_TORCH`)
- [ ] Build system / CMake / CI
- [ ] Python bindings / user-facing API

## Test Results on Supported Platforms

<!--
Per `CONTRIBUTING.md` §Pull Requests \#3, you MUST build and test on every
supported platform touched by this PR (or explain why a platform is not
reachable). Paste `pytest` output summaries below (a trimmed tail is fine —
include pass/fail counts, skipped tests, and any warnings).

If a platform was **not** tested, state the reason (e.g. "no hardware
available"). Reviewers may block the PR until coverage is provided or an
explicit owner signs off on a partial run.
-->

| Platform   | Built | `pytest` Result | Notes / Hardware |
| ---------- | :---: | --------------- | ---------------- |
| NVIDIA     |       |                 |                  |
| Iluvatar   |       |                 |                  |
| MetaX      |       |                 |                  |
| Cambricon  |       |                 |                  |
| Moore      |       |                 |                  |
| Ascend     |       |                 |                  |

<details>
<summary>Full `pytest` output (optional)</summary>

```text
paste here
```

</details>

## Benchmark / Performance Impact

<!--
Required for `perf` PRs; optional otherwise. Describe the benchmark harness,
shapes, dtypes, hardware, and include baseline vs. new numbers. If the PR is
not performance-sensitive, write "N/A".
-->

## Notes for Reviewers

<!--
Anything reviewers should focus on: subtle invariants, known trade-offs,
follow-up work intentionally left out of scope, etc.
-->

---

## Checklist

> Every contributor **must** verify every item below before requesting
> review. Tick each box only after the check has actually been performed —
> do not tick speculatively. If an item truly does not apply, replace the
> checkbox with `N/A` and briefly explain why in an inline comment.

### Title, Branch, and Commits

- [ ] PR **title** follows [Conventional Commits](https://www.conventionalcommits.org/) (e.g. `feat(nvidia): …`, `fix(cuda/gemm): …`).
- [ ] Branch name follows `<type>/xxx-yyyy-zzzz` where `<type>` matches the PR title's Conventional Commits type and words are joined with hyphens (see `CONTRIBUTING.md` §Branches).
- [ ] Each **commit** message follows Conventional Commits.
- [ ] Small PR is a **single squashable commit**; or, for a large PR, every commit is meaningful, well-formed, and independently reviewable (see `CONTRIBUTING.md` §Pull Requests \#1).
- [ ] No stray merge commits from `master` — the branch is rebased cleanly on top of the current `master`.
- [ ] No `fixup!` / `squash!` / `wip` commits remain.

### Scope and Design

- [ ] Changes are **minimal** — nothing unrelated to the stated motivation was added (`CONTRIBUTING.md` §Code/General \#1).
- [ ] No dead code, commented-out blocks, debug prints, `printf`/`std::cout`/`print(...)` left behind, or `TODO` without an owner and issue link.
- [ ] No unrelated formatting churn that would obscure the diff.
- [ ] Public API changes (if any) are intentional, documented, and reflected in affected callers/tests.

### General Code Hygiene (applies to all languages)

- [ ] The code is self-explanatory; comments were added **only** where the *why* is non-obvious (`CONTRIBUTING.md` §Code/General \#2).
- [ ] Every modified or added file **ends with a single trailing newline** (`CONTRIBUTING.md` §Code/General \#3).
- [ ] No trailing whitespace, tab/space mixing, or stray BOMs.
- [ ] Identifiers in comments and error messages are wrapped in backticks (e.g. ``the `seqlens_k` tensor``) (`CONTRIBUTING.md` §Code/General \#4).
- [ ] All comments and error messages are in **English** (`CONTRIBUTING.md` §Code/General \#5).
- [ ] Comments and error messages are complete sentences — capitalized first letter, terminal punctuation — **unless** the language/framework convention says otherwise (`CONTRIBUTING.md` §Code/General \#6; Python \#2).

### C++ Specific (if C++ files changed)

- [ ] Code follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) strictly.
- [ ] `clang-format` (version **21**, per `.github/workflows/clang-format.yml`) has been run against all modified `.h`, `.cc`, `.cuh`, and `.mlu` files; the diff is clean.
- [ ] `clang-tidy` concerns (per `.clang-tidy`) have been reviewed — no new warnings beyond the existing baseline.
- [ ] Operator parameter order is **inputs first, outputs last**; attributes are between inputs and outputs; naming follows PyTorch → ONNX → CUDA API precedence (`CONTRIBUTING.md` §C++ \#2).
- [ ] **No exceptions** are thrown. Error paths use `assert` with messages that include at least `__FILE__`, `__LINE__`, and `__func__` (`CONTRIBUTING.md` §C++ \#3).
- [ ] Error and warning message wording follows the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html#error-and-warning-messages) (`CONTRIBUTING.md` §C++ \#4).
- [ ] Kernel files are named correctly: custom = `kernel` / `kernel_v2` / …; well-known algorithms use the algorithm name; library-based implementations use the library name (`CONTRIBUTING.md` §C++ \#5).
- [ ] Kernel and kernel launcher are in **separate files**: launcher `.h`, kernel follows platform conventions (e.g. `.cuh` + `.cu`) even when non-templated (`CONTRIBUTING.md` §C++ \#6).
- [ ] Constructor **initializer list order matches member declaration order** (`CONTRIBUTING.md` §C++ \#7).
- [ ] Exactly **one blank line** between classes, between classes and functions, and between functions (`CONTRIBUTING.md` §C++ \#8).
- [ ] Exactly **one blank line** between members (functions *and* variables) within a class (`CONTRIBUTING.md` §C++ \#9).
- [ ] Exactly **one blank line** before and after the contents of a namespace (`CONTRIBUTING.md` §C++ \#10).
- [ ] New operators added via `src/base/<op>.h` (inheriting `Operator<Op>`) with platform implementations under `src/<category>/<platform>/` inheriting the base (`CONTRIBUTING.md` §Adding an Operator \#1–2).
- [ ] No raw `new`/`delete`; RAII / smart pointers / existing allocators are used.

### Python Specific (if Python files changed)

- [ ] Code is [PEP 8](https://peps.python.org/pep-0008/) compliant; `ruff check` passes cleanly on CI (see `.github/workflows/ruff.yml`).
- [ ] `ruff format --check` passes cleanly — if not, run `ruff format` and commit the result.
- [ ] Comments are complete English sentences, starting with a capital letter and ending with punctuation; Markdown backticks are used for code references (`CONTRIBUTING.md` §Python \#1).
- [ ] Framework-specific conventions (e.g. lowercase `pytest.skip` messages without terminal period) are honored where applicable (`CONTRIBUTING.md` §Python \#2).
- [ ] **No blank line** between the function signature and the body when there is no docstring or comment (`CONTRIBUTING.md` §Python \#3).
- [ ] A blank line is present **before and after** `if`, `for`, and similar control-flow statements (`CONTRIBUTING.md` §Python \#4).
- [ ] A blank line appears **before** each `return`, except when it directly follows a control-flow statement (`CONTRIBUTING.md` §Python \#5).
- [ ] Docstrings (if any) follow [PEP 257](https://peps.python.org/pep-0257/) (`CONTRIBUTING.md` §Python \#6).
- [ ] Type hints are added / kept consistent with the surrounding code.

### Testing

- [ ] `pytest` was run locally on **every supported platform** that this PR can affect, and the results are recorded in the "Test Results" table above (`CONTRIBUTING.md` §Pull Requests \#3).
- [ ] For any platform that could not be tested, an explicit reason is given in the table and a reviewer with access has been tagged.
- [ ] New functionality has matching tests under `tests/` following `tests/test_add.py` / `tests/test_gemm.py` patterns (`CONTRIBUTING.md` §Adding an Operator \#3).
- [ ] Tests use `pytest.mark.parametrize` correctly: dependent parameters share one decorator (e.g. `@pytest.mark.parametrize("dtype, rtol, atol", …)`), independent parameters use separate decorators ordered by parameter declaration.
- [ ] Where appropriate, `pytest.mark.auto_act_and_assert` is used and the test returns a `Payload` whose `func` and `ref` share the same calling convention.
- [ ] Default `dtype` / `device` parameterization is relied on, or overridden with an explicit `pytest.mark.parametrize` when necessary.
- [ ] Any new test that is flaky under parallelism is marked so, or documented to require `pytest -n 1`.
- [ ] For bug fixes: a regression test has been added that fails on `master` and passes with this PR.

### Build, CI, and Tooling

- [ ] The project builds cleanly from a fresh directory with `pip install .[dev]` on at least one affected platform.
- [ ] `compile_commands.json` still regenerates (CMake option `CMAKE_EXPORT_COMPILE_COMMANDS=ON` in `pyproject.toml` — required by the `code-lint` skill and `clang-tidy -p`).
- [ ] New backends / devices have been added to auto-detection in `CMakeLists.txt` under `if(AUTO_DETECT_DEVICES)` **and** to `if(AUTO_DETECT_BACKENDS)` if applicable.
- [ ] Only one CUDA-like GPU backend is selectable at a time — the existing mutual-exclusion check in `CMakeLists.txt` is not broken.
- [ ] Both CI workflows (`clang-format.yml`, `ruff.yml`) are green locally (or expected to be green on CI).
- [ ] No new runtime dependency was added without updating `pyproject.toml`'s `[project.optional-dependencies]` (or justified in the PR description).

### Documentation

- [ ] `README.md`, `CONTRIBUTING.md`, or inline docs updated when behavior, build flags, or developer workflow changed.
- [ ] New operators, new dispatch helpers, or new public utilities are documented (docstring, header comment, or an addition to `CONTRIBUTING.md` §Some Code Explanations).
- [ ] Any user-visible breaking change is called out explicitly under "Motivation" **and** in the commit/PR title with a `!` or `BREAKING CHANGE:` footer.

### Security and Safety

- [ ] No secrets, access tokens, internal URLs, customer data, or personal hardware identifiers have been committed.
- [ ] Third-party code is license-compatible and attributed.
- [ ] No unsafe pointer arithmetic, uninitialized reads, or missing bounds checks were introduced.
