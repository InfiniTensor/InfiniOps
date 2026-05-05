"""Create an integration branch from multiple independent base-op branches."""

import argparse
import subprocess


def _run(args: list[str]) -> None:
    print("+", " ".join(args))
    subprocess.run(args, check=True)


def _branch_ref(branch: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", branch],
        check=False,
        stdout=subprocess.DEVNULL,
    )

    if result.returncode == 0:
        return branch

    remote_ref = f"origin/{branch}"
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", remote_ref],
        check=False,
        stdout=subprocess.DEVNULL,
    )

    if result.returncode == 0:
        return remote_ref

    _run(["git", "fetch", "origin", branch])

    return remote_ref


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branches", nargs="+", help="Branches to integrate.")
    parser.add_argument(
        "--base",
        default="feat/torch-codegen",
        help="Base branch for the integration branch.",
    )
    parser.add_argument(
        "--target",
        default="codex/integrate-base-branches",
        help="Integration branch to create or reset.",
    )
    parser.add_argument(
        "--strategy",
        choices=("merge", "cherry-pick"),
        default="merge",
        help="How to apply each branch.",
    )
    parser.add_argument(
        "--reset-target",
        action="store_true",
        help="Reset the target branch if it already exists.",
    )

    args = parser.parse_args()

    switch_flag = "--force-create" if args.reset_target else "--create"
    _run(["git", "switch", switch_flag, args.target, _branch_ref(args.base)])

    for branch in args.branches:
        ref = _branch_ref(branch)

        if args.strategy == "merge":
            _run(["git", "merge", "--no-edit", ref])
        else:
            _run(["git", "cherry-pick", ref])


if __name__ == "__main__":
    main()
