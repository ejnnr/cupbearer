#!/bin/bash
#SBATCH --job-name=train_base
# Avoid cluttering the root directory with log files:
#SBATCH --output=slurm/%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --gres=shard:8
#SBATCH --time=01:00:00

# Usage:
# train_base.sh (clean|pixel_backdoor|noise_backdoor) (mnist|cifar10) (mlp|cnn)

set -euo pipefail

cd /nas/ucb/erik/abstractions
source .venv/bin/activate

MODE=$1
DATASET=$2
MODEL=$3
SEED=0

if [[ $MODE == "pixel_backdoor" ]]; then
    OPTIONS="+data@train_data=pixel_backdoor train_data.pixel_backdoor.p_backdoor=0.05"
elif [[ $MODE == "noise_backdoor" ]]; then
    OPTIONS="+data@train_data=noise_backdoor train_data.noise_backdoor.p_backdoor=0.2"
elif [[ $MODE == "clean" ]]; then
    OPTIONS=""
else
    echo "Unknown mode: $MODE"
    exit 1
fi

srun python -m abstractions.train_base \
    +data@train_data=$DATASET \
    $OPTIONS \
    seed=$SEED \
    dir.run=$MODE/$DATASET/$MODEL/$SEED

