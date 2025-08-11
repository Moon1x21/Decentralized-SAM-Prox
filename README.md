# Postech 2023-2 Deeplearning Final Project

## Environment Setup
Requisite packages can be installed directly from the `requirements.txt`.
```
pip install -r requirements.txt
```

or you can create a new environment from the `dec.yaml`.
```
conda env create -f dec.yaml
```

## Example of usage

#### Train ResNet-18 on CIFAR-10 using DFedSAM with 128 local batch sizes with MGS.

```
python main.py --dataset_name "CIFAR10" --image_size 32 --batch_size 128 --mode "meshgrid" --size 32 --lr 0.1 --model "ResNet18" --early_stop #total_iteration_num --local_iter #local_iteration_num --warmup_step 5  --multiple_local_gossip --optimization 'sam'

python main.py --dataset_name "CIFAR10" --image_size 32 --batch_size 128 --mode "ring" --size 32 --lr 0.1 --model "ResNet18" --early_stop #total_iteration_num --local_iter #local_iteration_num --warmup_step 5  --multiple_local_gossip --optimization 'sam'  

python main.py --dataset_name "CIFAR10" --image_size 32 --batch_size 128 --mode "all" --size 32 --lr 0.1 --model "ResNet18" --early_stop #total_iteration_num --local_iter #local_iteration_num --warmup_step 5  --multiple_local_gossip --optimization 'sam' 
```


#### Train ResNet-18 on CIFAR-10 using DFedSAM with 128 local batch sizes without MGS.

```
python main.py --dataset_name "CIFAR10" --image_size 32 --batch_size 128 --mode "meshgrid" --size 32 --lr 0.1 --model "ResNet18" --early_stop #total_iteration_num --local_iter #local_iteration_num --warmup_step 5 --optimization 'sam'

python main.py --dataset_name "CIFAR10" --image_size 32 --batch_size 128 --mode "ring" --size 32 --lr 0.1 --model "ResNet18" --early_stop #total_iteration_num --local_iter #local_iteration_num --warmup_step 5  --optimization 'sam'

python main.py --dataset_name "CIFAR10" --image_size 32 --batch_size 128 --mode "all" --size 32 --lr 0.1 --model "ResNet18" --early_stop #total_iteration_num --local_iter #local_iteration_num --warmup_step 5  --optimization 'sam' 
```

#### Train ResNet-18 on CIFAR-10 using OURS with 128 local batch sizes.

```
python main.py --dataset_name "CIFAR10" --image_size 32 --batch_size 128 --mode "meshgrid" --size 32 --lr 0.1 --model "ResNet18" --early_stop #total_iteration_num --local_iter #local_iteration_num --warmup_step 5  --multiple_local_gossip --optimization 'samprox' --mu #mu --gossip_step #gossip_step_num

python main.py --dataset_name "CIFAR10" --image_size 32 --batch_size 128 --mode "ring" --size 32 --lr 0.1 --model "ResNet18" --early_stop #total_iteration_num --local_iter #local_iteration_num --warmup_step 5  --multiple_local_gossip --optimization 'samprox' --mu #mu --gossip_step #gossip_step_num

python main.py --dataset_name "CIFAR10" --image_size 32 --batch_size 128 --mode "all" --size 32 --lr 0.1 --model "ResNet18" --early_stop #total_iteration_num --local_iter #local_iteration_num --warmup_step 5  --multiple_local_gossip --optimization 'samprox' --mu #mu --gossip_step #gossip_step_num
```

## Main argument explaination

```
--dataset_name : Datedset name ('CIFAR10','CIFAR100','TinyImageNet')
--image_size : data image size (int)
--size : clint number
--mode : topology type ('csgd', 'ring', 'meshgrid', 'exponential','all')
--early_stop : total iteration. local iteration * global iteration (int)
--multiple_local_gossip : Using MGS
--optimization: optimization type ('base', 'sam', 'samprox')
--mu: regualzation hyperparameter (int)
--local_iter: number of local iteration (int)
--gossip_step : step size of processing MGS. (int)
--rho: SAM rho hyperparameter (float)
```

This code is based on [Decentralized SGD and Average-direction SAM are Asymptotically Equivalent](https://github.com/Raiden-Zhu/ICML-2023-DSGD-and-SAM)
