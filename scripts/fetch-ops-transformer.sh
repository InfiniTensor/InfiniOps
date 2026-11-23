#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SUBMODULE_REL=third_party/ops-transformer
SUBMODULE="$ROOT/$SUBMODULE_REL"
GIT="git -c safe.directory=*"
EXPECTED=$($GIT -C "$ROOT" ls-tree HEAD "$SUBMODULE_REL" | awk '{print $3}')

if ! $GIT -C "$SUBMODULE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    $GIT -C "$ROOT" submodule update --init --checkout --force "$SUBMODULE_REL"
else
    ACTUAL=$($GIT -C "$SUBMODULE" rev-parse HEAD)
    if [ "$ACTUAL" != "$EXPECTED" ]; then
        $GIT -C "$ROOT" submodule update --init --checkout --force "$SUBMODULE_REL"
    fi
fi

printf 'ops-transformer: %s\n' "$($GIT -C "$SUBMODULE" rev-parse HEAD)"
