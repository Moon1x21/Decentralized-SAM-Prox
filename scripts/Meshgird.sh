
# CIFAR10 training 50000
# 50000/(512*16) = 6.10
# 50000/(64*16)  = 48.8


## ResNet18 + SAM + MGS
python main.py --dataset_name "CIFAR10" --image_size 32 --batch_size 128 --mode "meshgrid" --size 32 --lr 0.1 --model "ResNet18" --early_stop 5000 --local_iter 5 --warmup_step 5  --multiple_local_gossip --optimization 'sam'
python main.py --dataset_name "CIFAR100" --image_size 32 --batch_size 128 --mode "meshgrid" --size 32 --lr 0.1 --model "ResNet18" --early_stop 5000 --local_iter 5 --warmup_step 5  --multiple_local_gossip --optimization 'sam'

## ResNet18 + SAM 
python main.py --dataset_name "CIFAR10" --image_size 32 --batch_size 128 --mode "meshgrid" --size 32 --lr 0.1 --model "ResNet18" --early_stop 5000 --local_iter 5 --warmup_step 5  --optimization 'sam'
python main.py --dataset_name "CIFAR100" --image_size 32 --batch_size 128 --mode "meshgrid" --size 32 --lr 0.1 --model "ResNet18" --early_stop 5000 --local_iter 5 --warmup_step 5  --optimization 'sam'

## ResNet18 + OURS
python main.py --dataset_name "CIFAR10" --image_size 32 --batch_size 128 --mode "meshgrid" --size 32 --lr 0.1 --model "ResNet18" --early_stop 5000 --local_iter 5 --warmup_step 5  --multiple_local_gossip --optimization 'samprox' --mu 0.5 --gossip_step 1500
python main.py --dataset_name "CIFAR100" --image_size 32 --batch_size 128 --mode "meshgrid" --size 32 --lr 0.1 --model "ResNet18" --early_stop 5000 --local_iter 5 --warmup_step 5  --multiple_local_gossip --optimization 'samprox' --mu 0.5 --gossip_step 1500
