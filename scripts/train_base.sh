#!/bin/bash
#SBATCH --job-name=train_base
# Avoid cluttering the root directory with log files:
#SBATCH --output=slurm/%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --gres=shard:8
#SBATCH --time=01:00:00

# Usage:
# train_base.sh (clean|pixel_backdoor|noise_backdoor) (mnist_cnn|mnist_mlp|cifar10)

set -euo pipefail

# We're using 8 GPU shards, which correspond to 8192 MiB. We need to make sure Jax
# doesn't try to allocate more memory than that. More specifically, we allocate 90%
# (which is the Jax default). Without this, Jax would allocate 90% of the entire GPU.
TOTAL_MEMORY=$(nvidia-smi -i 0 --format=csv,noheader,nounits --query-gpu=memory.total)
TARGET_MEMORY=$(echo "8 * 1024 * 0.9" | bc -l)
FRACTION=$(echo "scale=2;$TARGET_MEMORY/$TOTAL_MEMORY" | bc)
echo "Allocating $FRACTION of GPU ($TARGET_MEMORY MiB out of $TOTAL_MEMORY MiB)"
export XLA_PYTHON_CLIENT_MEM_FRACTION=$FRACTION

# Initialize pyenv
# export PYENV_ROOT=/nas/ucb/erik/.pyenv
# command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)"
eval "$(/nas/ucb/erik/miniconda3/bin/conda shell.bash hook)"
conda activate abstractions

# This will also activate the virtual environment
cd /nas/ucb/erik/abstractions

MODE=$1
EXPERIMENT=$2
OVERRIDE=${3:-""}
SEED=${SLURM_ARRAY_TASK_ID:-0}

if [[ $MODE == "clean" ]]; then
    EXPERIMENT_OPTIONS=""
else
    EXPERIMENT_OPTIONS="$MODE"
fi

if [[ $OVERRIDE == "override" ]]; then
    OPTIONS="+override_output=True"
elif [[ $OVERRIDE == "" ]]; then
    OPTIONS=""
else
    echo "Invalid override option: $OVERRIDE"
    exit 1
fi


srun python -m abstractions.train_base \
    +experiment=["$EXPERIMENT","$EXPERIMENT_OPTIONS"] \
    seed=$SEED \
    dir.run=$MODE/$EXPERIMENT/$SEED \
    dir.log=results/base \
    $OPTIONS

