#!/bin/bash
# Allocate swap space to prevent OOM during full WGS runs.
#
# DeepVariant make_examples uses ~1.5-2 GB per shard for full WGS.
# On machines where (shards × 2 GB) exceeds physical RAM, the kernel
# OOM-kills the process. Adding swap lets the OS page out inactive
# data instead of killing the process.
#
# NVMe swap overhead is negligible: ~100µs per page fault, and only
# a small fraction of pages are actively swapped at any time.
# Measured overhead: <30 seconds across a full WGS run.
#
# Usage:
#   sudo bash scripts/setup_swap.sh          # auto-detect size
#   sudo bash scripts/setup_swap.sh --size 16  # explicit 16 GB
#   sudo bash scripts/setup_swap.sh --off      # disable and remove

set -euo pipefail

SWAP_FILE="/swapfile"

# --- Parse arguments ---
SWAP_SIZE_GB=""
DISABLE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size)
      SWAP_SIZE_GB="$2"
      shift 2
      ;;
    --off)
      DISABLE=true
      shift
      ;;
    -h|--help)
      echo "Usage: sudo bash $0 [--size GB] [--off]"
      echo ""
      echo "Options:"
      echo "  --size GB   Set swap size in GB (default: auto-detect)"
      echo "  --off       Disable swap and remove swapfile"
      echo ""
      echo "Auto-detection formula:"
      echo "  swap_gb = max(0, nproc × 2 - physical_ram_gb + 4)"
      echo "  Ensures enough headroom for nproc shards at ~2 GB each."
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# --- Disable mode ---
if [[ "${DISABLE}" == "true" ]]; then
  if swapon --show | grep -q "${SWAP_FILE}"; then
    swapoff "${SWAP_FILE}"
    echo "Swap disabled: ${SWAP_FILE}"
  fi
  if [[ -f "${SWAP_FILE}" ]]; then
    rm -f "${SWAP_FILE}"
    echo "Swapfile removed: ${SWAP_FILE}"
  else
    echo "No swapfile found at ${SWAP_FILE}"
  fi
  exit 0
fi

# --- Check if already active ---
if swapon --show | grep -q "${SWAP_FILE}"; then
  _current_gb=$(swapon --show --bytes | awk -v f="${SWAP_FILE}" '$1==f {printf "%d", $3/1024/1024/1024}')
  echo "Swap already active: ${SWAP_FILE} (${_current_gb} GB)"
  exit 0
fi

# --- Auto-detect swap size ---
if [[ -z "${SWAP_SIZE_GB}" ]]; then
  _nproc=$(nproc)
  _ram_gb=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
  # Each WGS shard needs ~2 GB. Add 4 GB headroom for OS + Docker.
  _needed=$(( _nproc * 2 + 4 ))
  SWAP_SIZE_GB=$(( _needed - _ram_gb ))
  if [[ "${SWAP_SIZE_GB}" -le 0 ]]; then
    echo "No swap needed: ${_ram_gb} GB RAM is sufficient for ${_nproc} shards (need ~${_needed} GB)"
    exit 0
  fi
  # Cap at 32 GB — more than enough for any reasonable config.
  [[ "${SWAP_SIZE_GB}" -gt 32 ]] && SWAP_SIZE_GB=32
  echo "Auto-detected: ${_nproc} CPUs × 2 GB/shard + 4 GB headroom = ${_needed} GB needed"
  echo "Physical RAM: ${_ram_gb} GB → adding ${SWAP_SIZE_GB} GB swap"
fi

# --- Check disk space ---
_disk_avail_gb=$(df -BG / | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
if [[ "${_disk_avail_gb}" -lt $(( SWAP_SIZE_GB + 5 )) ]]; then
  echo "ERROR: Only ${_disk_avail_gb} GB disk available. Need ${SWAP_SIZE_GB} GB for swap + 5 GB headroom." >&2
  exit 1
fi

# --- Create swapfile ---
echo "Creating ${SWAP_SIZE_GB} GB swapfile at ${SWAP_FILE}..."
# Remove stale swapfile if it exists but isn't active.
[[ -f "${SWAP_FILE}" ]] && rm -f "${SWAP_FILE}"

fallocate -l "${SWAP_SIZE_GB}G" "${SWAP_FILE}"
chmod 600 "${SWAP_FILE}"
mkswap "${SWAP_FILE}" > /dev/null
swapon "${SWAP_FILE}"

# Low swappiness — only use swap under real memory pressure.
sysctl -q vm.swappiness=10

echo "Swap enabled: ${SWAP_SIZE_GB} GB at ${SWAP_FILE} (swappiness=10)"
free -h | head -3
