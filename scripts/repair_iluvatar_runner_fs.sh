#!/usr/bin/env bash

# Diagnose and repair the Iluvatar GitHub runner workspace when actions fail with:
# "Structure needs cleaning".
#
# Default mode is safe for an SSH session:
#   - Stop the GitHub runner.
#   - Kill stale runner processes.
#   - Try to remove broken GitHub Actions caches.
#   - Restart the runner if cleanup succeeds.
#
# Filesystem repair is intentionally gated behind --repair because it may require
# unmounting /home and can break active SSH sessions.

set -u
set -o pipefail

RUNNER_DIR="${RUNNER_DIR:-/home/zkjh/actions-runner}"
RUNNER_USER="${RUNNER_USER:-zkjh}"
RUNNER_GROUP="${RUNNER_GROUP:-zkjh}"
SERVICE_NAME="${SERVICE_NAME:-actions.runner.InfiniTensor-InfiniOps.Iluvatar-Server-001.service}"
TARGET_PATH="${TARGET_PATH:-${RUNNER_DIR}/_work/_actions/actions/upload-artifact/v4/.licenses/npm}"
MODE="cleanup"
ALLOW_UNMOUNT="${ALLOW_UNMOUNT:-0}"

if [ "${1:-}" = "--repair" ]; then
  MODE="repair"
elif [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  cat <<EOF
Usage:
  bash $0
      Safe online cleanup and runner restart.

  ALLOW_UNMOUNT=1 bash $0 --repair
      Stop runner, unmount the filesystem containing ${RUNNER_DIR}, repair it,
      remount it, clean runner caches, and restart runner.

Environment overrides:
  RUNNER_DIR=${RUNNER_DIR}
  RUNNER_USER=${RUNNER_USER}
  RUNNER_GROUP=${RUNNER_GROUP}
  SERVICE_NAME=${SERVICE_NAME}
  TARGET_PATH=${TARGET_PATH}
  ALLOW_UNMOUNT=${ALLOW_UNMOUNT}
EOF
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

warn() {
  printf '[%s] WARN: %s\n' "$(date '+%F %T')" "$*" >&2
}

die() {
  printf '[%s] ERROR: %s\n' "$(date '+%F %T')" "$*" >&2
  exit 1
}

require_root() {
  if [ "$(id -u)" -ne 0 ]; then
    die "Run as root."
  fi
}

service_exists() {
  systemctl list-unit-files "${SERVICE_NAME}" >/dev/null 2>&1
}

stop_runner() {
  log "Stopping runner service/processes."
  if service_exists; then
    systemctl stop "${SERVICE_NAME}" || warn "systemctl stop failed: ${SERVICE_NAME}"
  else
    warn "Service not found: ${SERVICE_NAME}; killing runner processes only."
  fi

  pkill -f "${RUNNER_DIR}/.*Runner.Worker" 2>/dev/null || true
  pkill -f "${RUNNER_DIR}/.*Runner.Listener" 2>/dev/null || true
  sleep 2
}

start_runner() {
  log "Starting runner."
  if service_exists; then
    systemctl start "${SERVICE_NAME}" || warn "systemctl start failed: ${SERVICE_NAME}"
    systemctl status "${SERVICE_NAME}" --no-pager || true
    return
  fi

  if [ -x "${RUNNER_DIR}/run.sh" ]; then
    su - "${RUNNER_USER}" -c "cd '${RUNNER_DIR}' && nohup ./run.sh > runner.log 2>&1 &" \
      || warn "fallback ./run.sh start failed"
  else
    warn "No service and no executable ${RUNNER_DIR}/run.sh; runner not started."
  fi
}

print_diagnostics() {
  log "Runner directory: ${RUNNER_DIR}"
  log "Target path: ${TARGET_PATH}"
  log "Mount information:"
  findmnt -T "${RUNNER_DIR}" -o SOURCE,FSTYPE,TARGET,OPTIONS || true
  df -hT "${RUNNER_DIR}" || true

  log "Recent filesystem/kernel messages:"
  dmesg -T 2>/dev/null | tail -n 160 | grep -Ei 'xfs|ext4|error|corrupt|clean|I/O|metadata|EUCLEAN' || true
}

cleanup_runner_cache() {
  local log_file
  log_file="$(mktemp /tmp/iluvatar-runner-cleanup.XXXXXX.log)"

  log "Removing GitHub Actions cache directories. Log: ${log_file}"
  rm -rf \
    "${RUNNER_DIR}/_work/_actions/actions/upload-artifact" \
    "${RUNNER_DIR}/_work/_actions/actions/cache" \
    "${RUNNER_DIR}/_work/_actions/actions/checkout" \
    >"${log_file}" 2>&1
  local rc=$?

  if [ "${rc}" -ne 0 ]; then
    warn "Cache cleanup failed with exit code ${rc}."
    cat "${log_file}" >&2 || true
    if grep -qi 'Structure needs cleaning' "${log_file}"; then
      warn "Detected filesystem EUCLEAN: Structure needs cleaning."
      return 70
    fi
    return "${rc}"
  fi

  if [ -d "${RUNNER_DIR}/_work" ]; then
    chown -R "${RUNNER_USER}:${RUNNER_GROUP}" "${RUNNER_DIR}/_work" \
      || warn "chown failed for ${RUNNER_DIR}/_work"
  fi
  log "Cache cleanup succeeded."
  return 0
}

readonly_fs_check() {
  local dev="$1"
  local fstype="$2"

  log "Running read-only filesystem check for ${dev} (${fstype})."
  case "${fstype}" in
    xfs)
      command -v xfs_repair >/dev/null 2>&1 || die "xfs_repair not found. Install xfsprogs."
      xfs_repair -n "${dev}"
      ;;
    ext2|ext3|ext4)
      command -v fsck.ext4 >/dev/null 2>&1 || command -v fsck >/dev/null 2>&1 || die "fsck not found."
      fsck.ext4 -fn "${dev}" 2>/dev/null || fsck -fn "${dev}"
      ;;
    *)
      die "Unsupported filesystem type: ${fstype}"
      ;;
  esac
}

repair_filesystem() {
  local dev="$1"
  local fstype="$2"
  local mnt="$3"

  if [ "${ALLOW_UNMOUNT}" != "1" ]; then
    cat >&2 <<EOF

Refusing to unmount automatically.

Detected:
  device: ${dev}
  fstype: ${fstype}
  mount:  ${mnt}

To run the destructive repair step, use a maintenance shell where unmounting ${mnt}
is acceptable, then run:

  ALLOW_UNMOUNT=1 bash $0 --repair

If ${mnt} is /, /home, or contains your active SSH session, prefer rescue/single-user
mode instead of running this from a normal SSH login.
EOF
    exit 2
  fi

  if [ "${mnt}" = "/" ]; then
    die "Refusing to unmount /. Boot into rescue mode and run filesystem repair there."
  fi

  log "Processes using ${mnt}:"
  fuser -vm "${mnt}" || true

  log "Unmounting ${mnt}."
  umount "${mnt}" || die "umount failed for ${mnt}. Stop users/processes or use rescue mode."

  log "Repairing ${dev} (${fstype})."
  case "${fstype}" in
    xfs)
      command -v xfs_repair >/dev/null 2>&1 || die "xfs_repair not found. Install xfsprogs."
      xfs_repair "${dev}" || die "xfs_repair failed."
      ;;
    ext2|ext3|ext4)
      command -v fsck.ext4 >/dev/null 2>&1 || command -v fsck >/dev/null 2>&1 || die "fsck not found."
      fsck.ext4 -f -y "${dev}" 2>/dev/null || fsck -f -y "${dev}" || die "fsck failed."
      ;;
    *)
      die "Unsupported filesystem type: ${fstype}"
      ;;
  esac

  log "Mounting ${mnt}."
  mount "${mnt}" || die "mount failed for ${mnt}."
}

main() {
  require_root

  if [ ! -d "${RUNNER_DIR}" ]; then
    die "Runner directory does not exist: ${RUNNER_DIR}"
  fi

  print_diagnostics
  stop_runner

  local dev fstype mnt
  dev="$(findmnt -T "${RUNNER_DIR}" -n -o SOURCE || true)"
  fstype="$(findmnt -T "${RUNNER_DIR}" -n -o FSTYPE || true)"
  mnt="$(findmnt -T "${RUNNER_DIR}" -n -o TARGET || true)"

  [ -n "${dev}" ] || die "Could not detect backing device for ${RUNNER_DIR}."
  [ -n "${fstype}" ] || die "Could not detect filesystem type for ${RUNNER_DIR}."
  [ -n "${mnt}" ] || die "Could not detect mount point for ${RUNNER_DIR}."

  if [ "${MODE}" = "repair" ]; then
    readonly_fs_check "${dev}" "${fstype}" || true
    repair_filesystem "${dev}" "${fstype}" "${mnt}"
    cleanup_runner_cache || true
    start_runner
    exit 0
  fi

  cleanup_runner_cache
  local cleanup_rc=$?
  if [ "${cleanup_rc}" -eq 0 ]; then
    start_runner
    exit 0
  fi

  readonly_fs_check "${dev}" "${fstype}" || true
  cat >&2 <<EOF

Runner cache cleanup failed. This is likely a real filesystem issue.

Detected:
  device: ${dev}
  fstype: ${fstype}
  mount:  ${mnt}

Next step, in a maintenance/rescue shell:
  ALLOW_UNMOUNT=1 bash $0 --repair

Do not keep rerunning CI on this runner until the filesystem is repaired.
EOF
  exit "${cleanup_rc}"
}

main "$@"
