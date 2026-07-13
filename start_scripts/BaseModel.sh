echo $RESOURCE_NUM_GPU
echo $DISTRIBUTED_NODE_COUNT
echo $DISTRIBUTED_NODE_RANK
echo $DISTRIBUTED_MASTER_HOSTS
echo $DISTRIBUTED_PYTORCH_PORT

GLOBAL_NUM_PROCESSES=$(($RESOURCE_NUM_GPU * $DISTRIBUTED_NODE_COUNT))
echo "Launching training with $GLOBAL_NUM_PROCESSES total processes"

NETWORK_INTERFACE="eth0"
if ip addr | grep -q "ens3"; then
    NETWORK_INTERFACE="ens3"
elif ip addr | grep -q "eno1"; then
    NETWORK_INTERFACE="eno1"
fi
echo "[INFO] Using network interface: $NETWORK_INTERFACE"

export PYTORCH_DISABLE_IPV6=1
export NCCL_SHM_DISABLE=0
export NCCL_P2P_DISABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export GLOO_SOCKET_IFNAME=$NETWORK_INTERFACE
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($RESOURCE_NUM_GPU - 1)))

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

cd "$PROJECT_ROOT"
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

echo "[INFO] Starting accelerate launch..."

accelerate launch \
    --multi_gpu \
    --num_processes=$GLOBAL_NUM_PROCESSES \
    --num_machines=$DISTRIBUTED_NODE_COUNT \
    --machine_rank=$DISTRIBUTED_NODE_RANK \
    --main_process_ip=$DISTRIBUTED_MASTER_HOSTS \
    --main_process_port=$LUBAN_AVAILBLE_PORT_0 \
    "${PROJECT_ROOT}/foundationgeo/scripts/train_base.py" \
        --config "${PROJECT_ROOT}/configs/train/BaseModel.json" \
        --workspace "${PROJECT_ROOT}/workspace/FoundationGeo_BaseModel" \
        --gradient_accumulation_steps 1 \
        --batch_size_forward 8 \
        --enable_gradient_checkpointing False \
        --checkpoint latest \
        --vis_every 500 \
        --enable_mlflow True \
        --save_every 1000 \
        --log_every 200 \
        --num_iterations 1000000 \
        --enable_mixed_precision False
