#!/bin/bash
#SBATCH --job-name 'Scheduling_all'
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1  
#SBATCH --time 2-0
#SBATCH --partition debug
#SBATCH -o /home/junemun/code/DSAMConsensus/logs/Deeplearning/slurm-%A-%x.out
#SBATCH -e /home/junemun/code/DSAMConsensus/errs/%x_%j.err

echo "activate dsgd"
#module load anaconda3/2022.05
module load cuda/11.6.2


echo "activate dec"
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate dec

echo "Start Training"

## ResNet18 + OURS 
python main.py --dataset_name "CIFAR10" --image_size 32 --batch_size 128 --mode "all" --size 32 --lr 0.1 --model "ResNet18" --early_stop 5000 --local_iter 5 --warmup_step 5  --optimization 'samprox' --mu 1.0 --wd 0.0005 --gamma 0.998 --sam_scheduler 'log'
python main.py --dataset_name "CIFAR10" --image_size 32 --batch_size 128 --mode "all" --size 32 --lr 0.1 --model "ResNet18" --early_stop 5000 --local_iter 5 --warmup_step 5  --optimization 'samprox' --mu 1.0 --wd 0.0005 --gamma 0.998 --sam_scheduler 'step'
