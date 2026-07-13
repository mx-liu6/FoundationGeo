#!/usr/bin/env bash
set -euo pipefail

NETWORK_INTERFACE="eth0"
if ip addr | grep -q "ens3"; then
    NETWORK_INTERFACE="ens3"
elif ip addr | grep -q "eno1"; then
    NETWORK_INTERFACE="eno1"
fi
echo "[INFO] Using network interface: ${NETWORK_INTERFACE}"

export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=${NETWORK_INTERFACE}
export GLOO_SOCKET_IFNAME=${NETWORK_INTERFACE}
export NCCL_DEBUG=WARN

FOUNDATIONGEO_ROOT="${FOUNDATIONGEO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

CONFIG_PATH="${FOUNDATIONGEO_ROOT}/configs/eval/all_benchmarks.json"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${FOUNDATIONGEO_ROOT}/checkpoints/FoundationGeo_v1.pt}"
OUTPUT_DIR="${FOUNDATIONGEO_ROOT}/eval_output/FoundationGeo"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/FoundationGeo_v1_res9_fp32.json}"
EVAL_SCRIPT="${FOUNDATIONGEO_ROOT}/foundationgeo/scripts/eval_baseline.py"
BASELINE_SCRIPT="${FOUNDATIONGEO_ROOT}/baselines/foundationgeo.py"
RESOLUTION_LEVEL=9

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] Starting eval"
echo "[INFO] Config: ${CONFIG_PATH}"
echo "[INFO] Checkpoint: ${CHECKPOINT_PATH}"
echo "[INFO] Output dir: ${OUTPUT_DIR}"

cd "${FOUNDATIONGEO_ROOT}"
if [[ -n "${CONDA_SH:-}" ]]; then
    source "${CONDA_SH}"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
if [[ -n "${CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
    conda activate "${CONDA_ENV}"
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "[ERROR] Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

echo "[INFO] Running checkpoint ${CHECKPOINT_PATH}"
echo "[INFO] Output: ${OUTPUT_PATH}"

python "${EVAL_SCRIPT}" \
    --baseline "${BASELINE_SCRIPT}" \
    --config "${CONFIG_PATH}" \
    --output "${OUTPUT_PATH}" \
    --pretrained "${CHECKPOINT_PATH}" \
    --resolution_level "${RESOLUTION_LEVEL}" \
    --version v1
