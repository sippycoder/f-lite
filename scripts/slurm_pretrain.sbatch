#!/bin/bash

#SBATCH --job-name=castor_training
#SBATCH --output=slurm_logs/slurm-%x-%j.out # %x for job name, %j for job ID
#SBATCH --error=slurm_logs/slurm-%x-%j.err
# User will specify --nodes and --partition on sbatch command line
# e.g., sbatch --nodes=2 --partition=my_partition train.sbatch

#SBATCH --ntasks-per-node=1     # We run one launcher per node
#SBATCH --gpus-per-node=8       # Each launcher will manage 8 processes, one per GPU

# --- Project and Log Directories ---
PROJECT_DIR=${PROJECT_DIR:-"/fsx/ubuntu/workspace/repo/f-lite"}
LOG_DIR=${LOG_DIR:-"/fsx/ubuntu/workspace/repo/f-lite/logs"}

echo "Changing directory to Project Directory: ${PROJECT_DIR}"
cd "${PROJECT_DIR}" || { echo "Failed to cd into ${PROJECT_DIR}"; exit 1; }
echo "Current working directory: $(pwd)"

# --- User defined ENVs for AWS Hyperpod ---
export NCCL_PROTO="Simple"
export FI_PROVIDER="efa"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_EFA_USE_HUGE_PAGE="0"
export FI_EFA_SET_CUDA_SYNC_MEMOPS="0"
export NCCL_SOCKET_IFNAME="^docker,lo,veth,eth"
export LD_PRELOAD="/usr/local/cuda-12.8/lib/libnccl.so"

# --- Conda environment ---
CONDA_ENV_NAME="pollux"

CONDA_PATH=${CONDA_PATH:-"/fsx/ubuntu/miniconda3"}
export PATH="$CONDA_PATH/bin:$PATH"
source $CONDA_PATH/etc/profile.d/conda.sh

echo "Attempting to activate conda environment: ${CONDA_ENV_NAME}"
_CONDA_ROOT=$(conda info --base 2>/dev/null)

if [ -z "${_CONDA_ROOT}" ]; then
    echo "Error: conda command not found or conda base not determined."
    echo "Please ensure conda is installed and initialized."
    exit 1
fi

conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment: ${CONDA_ENV_NAME}"
    echo "Please ensure the environment exists and conda is correctly set up."
    exit 1
fi
echo "Conda environment ${CONDA_ENV_NAME} activated successfully."
echo "Python executable: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

# --- PyTorch distributed setup ---
# Determine Master Address and Port from Slurm
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500 # Default port

echo "--- Slurm Job Information ---"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "SLURM_NNODES: ${SLURM_NNODES}"
echo "SLURM_NTASKS_PER_NODE: ${SLURM_NTASKS_PER_NODE}"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "--- End Slurm Job Information ---"


AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

TORCHRUN_CMD="torchrun"

# TORCHRUN_ARGS:
# torchrun will use the PytorchMASTER_ADDR and PytorchMASTER_PORT for rendezvous.
# nnodes and node_rank are typically auto-detected by torchrun from Slurm environment variables.
declare -a TORCHRUN_ARGS=(
    "--nnodes=${SLURM_NNODES}"
    "--nproc_per_node=8"
    "--rdzv_backend=c10d"
    "--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
    "--log_dir=${LOG_DIR}/torchrun_logs/job_${SLURM_JOB_ID}_node_${SLURM_NODEID}" # Per-node torchrun logs
)

# Training script module and its arguments
declare -a TRAIN_SCRIPT_ARGS=(
    "-m"
    "f_lite.train"
)

BUCKET_NAME="train-bucket-1"

declare -a TRAINING_ARGS=(
    "--vae_path" "/fsx/ubuntu/workspace/checkpoints/FLUX.1-dev/vae"
    "--text_encoder_path" "/fsx/ubuntu/workspace/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct"
    "--processor_path" "/fsx/ubuntu/workspace/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct"
    "--model_width" "2048"
    "--model_depth" "24"
    "--model_head_dim" "256"
    "--rope_base" "10000"
    "--train_data_path" "${BUCKET_NAME}"
    "--base_image_dir" "/fsx/metadata/training"
    "--image_column" "media_path"
    "--caption_column" "captions"
    "--resolution" "256"
    "--center_crop"
    "--train_batch_size" "32"
    "--num_epochs" "1"
    "--max_steps" "1000000"
    "--gradient_accumulation_step" "1"
    "--learning_rate" "2e-4"
    "--weight_decay" "0.01"
    "--lr_scheduler" "wsd"
    "--num_warmup_steps" "4000"
    "--seed" "47"
    "--output_dir" "/fsx/checkpoints/f_lite/bs_512_${BUCKET_NAME}"
    "--checkpointing_steps" "5000"
    "--resume_from_checkpoint" "latest"
    "--mixed_precision" "bf16"
    "--report_to" "wandb"
    "--project_name" "ablations"
    "--run_name" "pretrain_bs_512_${BUCKET_NAME}"
    "--batch_multiplicity" "1"
    "--sample_every" "5000"
    "--sample_prompts_file" "prompts.txt"
    # "--debug"
)

echo "--- srun command execution ---"
echo "Starting training with ${SLURM_NNODES} nodes."
echo "Host where sbatch script is running: $(hostname)"
echo "User: $(whoami)"
echo "Current working directory: $(pwd)"

# The srun command structure requested by user.
# The -l flag labels srun output lines with the task number.
# srun will launch this command once per node (due to --ntasks-per-node=1).

echo "TORCHRUN_CMD: ${TORCHRUN_CMD}"
echo "TORCHRUN_ARGS: ${TORCHRUN_ARGS[*]}"
echo "TRAIN_SCRIPT_ARGS: ${TRAIN_SCRIPT_ARGS[*]}"
echo "TRAINING_ARGS: ${TRAINING_ARGS[*]}"

# Ensure all necessary variables are exported for srun tasks
export PATH FI_PROVIDER FI_EFA_USE_DEVICE_RDMA FI_EFA_USE_HUGE_PAGE FI_EFA_SET_CUDA_SYNC_MEMOPS NCCL_PROTO NCCL_SOCKET_IFNAME LD_PRELOAD MASTER_ADDR MASTER_PORT

srun ${AUTO_RESUME} \
    ${TORCHRUN_CMD} \
    ${TORCHRUN_ARGS[@]} \
    ${TRAIN_SCRIPT_ARGS[@]} \
    ${TRAINING_ARGS[@]}

EXIT_CODE=$?
echo "srun command finished with exit code ${EXIT_CODE}."

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Training job failed. Please check logs in slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out/err and any application specific logs."
fi

exit ${EXIT_CODE}
