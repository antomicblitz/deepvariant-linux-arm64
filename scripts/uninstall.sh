#!/bin/bash
# DeepVariant macOS ARM64 Uninstaller
#
# Removes DeepVariant binaries, Python environment, models, and shell
# profile entries created by install.sh.
#
# Usage:
#   $DEEPVARIANT_HOME/scripts/uninstall.sh
#   # or:
#   ~/.deepvariant/scripts/uninstall.sh

set -euo pipefail

DV_HOME="${DEEPVARIANT_HOME:-$HOME/.deepvariant}"
QUICKTEST_DIR="$HOME/deepvariant-quicktest"

echo "============================================"
echo " DeepVariant macOS ARM64 Uninstaller"
echo "============================================"
echo ""

if [[ ! -d "$DV_HOME" ]]; then
  echo "Nothing to uninstall — ${DV_HOME} does not exist."
  exit 0
fi

# Detect environment type
ENV_TYPE=""
CONDA_ENV_NAME=""
if [[ -f "$DV_HOME/.env_type" ]]; then
  ENV_INFO=$(cat "$DV_HOME/.env_type")
  if [[ "$ENV_INFO" == venv ]]; then
    ENV_TYPE="venv"
  elif [[ "$ENV_INFO" == conda:* ]]; then
    ENV_TYPE="conda"
    CONDA_ENV_NAME="${ENV_INFO#conda:}"
  fi
fi

# Show what will be removed
echo "The following will be removed:"
echo ""
echo "  1. ${DV_HOME}/"
DV_SIZE=$(du -sh "$DV_HOME" 2>/dev/null | cut -f1)
echo "     (binaries, models, scripts${ENV_TYPE:+, ${ENV_TYPE} environment}) — ${DV_SIZE}"

if [[ "$ENV_TYPE" == "conda" && -n "$CONDA_ENV_NAME" ]]; then
  echo "  2. Conda environment: ${CONDA_ENV_NAME}"
fi

SHELL_RC=""
MARKER="# DeepVariant macOS ARM64"
for rc in "$HOME/.zshrc" "$HOME/.bash_profile" "$HOME/.bashrc"; do
  if [[ -f "$rc" ]] && grep -q "$MARKER" "$rc" 2>/dev/null; then
    SHELL_RC="$rc"
    break
  fi
done

if [[ -n "$SHELL_RC" ]]; then
  STEP=$([[ "$ENV_TYPE" == "conda" ]] && echo "3" || echo "2")
  echo "  ${STEP}. Shell profile entries in ${SHELL_RC}"
fi

if [[ -d "$QUICKTEST_DIR" ]]; then
  QT_SIZE=$(du -sh "$QUICKTEST_DIR" 2>/dev/null | cut -f1)
  STEP=$([[ "$ENV_TYPE" == "conda" ]] && echo "4" || echo "3")
  echo "  ${STEP}. ${QUICKTEST_DIR}/ (quicktest data) — ${QT_SIZE}"
fi

echo ""
read -r -p "Proceed with uninstall? [y/N] " response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 0
fi

echo ""

# 1. Remove conda environment (before removing DV_HOME, which has .env_type)
if [[ "$ENV_TYPE" == "conda" && -n "$CONDA_ENV_NAME" ]]; then
  echo "--- Removing conda environment '${CONDA_ENV_NAME}'..."
  CONDA_CMD=""
  for cmd in conda mamba micromamba; do
    if command -v "$cmd" &>/dev/null; then
      CONDA_CMD="$cmd"
      break
    fi
  done

  if [[ -n "$CONDA_CMD" ]]; then
    if ${CONDA_CMD} env list 2>/dev/null | grep -qE "^${CONDA_ENV_NAME} "; then
      ${CONDA_CMD} env remove -y -n "$CONDA_ENV_NAME"
      echo "  Removed conda env '${CONDA_ENV_NAME}'."
    else
      echo "  Conda env '${CONDA_ENV_NAME}' not found (already removed?)."
    fi
  else
    echo "  WARNING: conda/mamba/micromamba not found. Remove the env manually:"
    echo "           conda env remove -n ${CONDA_ENV_NAME}"
  fi
fi

# 2. Remove install directory
echo "--- Removing ${DV_HOME}/..."
rm -rf "$DV_HOME"
echo "  Done."

# 3. Remove shell profile entries
if [[ -n "$SHELL_RC" ]]; then
  echo "--- Cleaning ${SHELL_RC}..."
  # Remove the 3-line block: marker, DEEPVARIANT_HOME export, PATH export,
  # plus the blank line before it
  TMPFILE=$(mktemp)
  awk -v marker="$MARKER" '
    BEGIN { skip = 0; blank = "" }
    {
      if ($0 ~ marker) {
        skip = 1
        blank = ""
        next
      }
      if (skip && /^export (DEEPVARIANT_HOME|PATH)=/) {
        next
      }
      if (skip && /^$/) {
        skip = 0
        next
      }
      skip = 0
      if (/^$/) {
        blank = blank $0 "\n"
      } else {
        printf "%s", blank
        blank = ""
        print
      }
    }
    END { }
  ' "$SHELL_RC" > "$TMPFILE"
  mv "$TMPFILE" "$SHELL_RC"
  echo "  Removed DeepVariant entries from ${SHELL_RC}."
fi

# 4. Remove quicktest data
if [[ -d "$QUICKTEST_DIR" ]]; then
  echo "--- Removing ${QUICKTEST_DIR}/..."
  rm -rf "$QUICKTEST_DIR"
  echo "  Done."
fi

echo ""
echo "============================================"
echo " DeepVariant has been uninstalled."
echo "============================================"
echo ""
echo "Open a new terminal to clear environment variables."
