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
SEED=${SLURM_ARRAY_TASK_ID:-0}

if [[ $MODE == "clean" ]]; then
    OPTIONS=""
else
    OPTIONS="$MODE"
fi

srun python -m abstractions.train_base \
    +experiment=["$EXPERIMENT","$OPTIONS"] \
    seed=$SEED \
    dir.run=$MODE/$EXPERIMENT/$SEED \
    dir.log=results/base

