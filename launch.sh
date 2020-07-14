#!/bin/bash

DEFAULT_RUN_PATH=/zfs1/jdurrant/durrantlab/hag63/leadopt_pytorch/

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <run_name> <gpu_partition> <**args>"
    exit 0
fi

ABS_SCRIPT=$(pwd)/train.py

# navigate to runs directory
RUNS_DIR="${RUNS_DIR:-$DEFAULT_RUN_PATH}"
cd $RUNS_DIR

if [[ -d $1 ]]; then
    echo "Warning: run directory $1 already exists!"
    exit -1
fi

echo "Creating run directory ($1)..."
mkdir $1

echo "Running script..."
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --output=$RUNS_DIR/$1/slurm_out.txt
#SBATCH --error=$RUNS_DIR/$1/slurm_err.txt
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cluster=gpu
#SBATCH --partition=$2
#SBATCH --mail-user=hag63@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --mem=80g

cd /ihome/jdurrant/hag63/cbio/leadopt_pytorch/
source ./setup.sh

PYTHON_PATH=$PYTHON_PATH:/ihome/jdurrant/hag63/cbio/leadopt_pytorch/

cd $RUNS_DIR/$1
python train.py $4
EOT
