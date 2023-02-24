#!/bin/bash
#SBATCH --job-name=eacikgoz17
#SBATCH --partition=ai
#SBATCH --qos=ai 
#SBATCH --account=ai 
##SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
##SBATCH --constraint=rtx_a6000
#SBATCH --constraint=tesla_t4
#SBATCH --mem=60G 
#SBATCH --time=7-0:0:0
#SBATCH --output=logs/mlp-2-gpu_%J.log
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=eacikgoz17@ku.edu.tr


echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

module load anaconda/3.6
source activate blind_robot
echo 'number of processors:'$(nproc)
nvidia-smi

python main.py

source deactivate