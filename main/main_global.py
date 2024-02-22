

import os
import copy
import torch
import socket
import datetime
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from datasets import load_dataset
from networks import load_model
from workers.worker_vision import *
from utils.scheduler import *
from utils.utils import *
from utils.minimizers import SAM

import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
dir_path = os.path.dirname(__file__)

nfs_dataset_path1 = '/mnt/nfs4-p1/ckx/datasets/'
nfs_dataset_path2 = '/nfs4-p1/ckx/datasets/'

# torch.set_num_threads(4) 

def main(args):
    set_seed(args)
    print(args)

    # check nfs dataset path
    if os.path.exists(nfs_dataset_path1):
        args.dataset_path = nfs_dataset_path1
    elif os.path.exists(nfs_dataset_path2):
        args.dataset_path = nfs_dataset_path2

    log_id = 'Stochastic' + args.dataset_name + '_sam_'+ args.mode 
    # if(args.multiple_local_gossip):
    #     log_name = args.optimization + '-' + args.mode + '-MGS_' + str(args.gossip_step)+'-local_'+str(args.local_iter)
    # else:
    #     log_name = args.optimization + '-' + args.mode +'-local_'+str(args.local_iter)
    if(args.multiple_local_gossip):
        log_name = args.optimization+'_MGS_' + str(args.early_stop/args.local_iter)
    else:
        log_name = args.optimization + str(args.early_stop/args.local_iter)
        
    wandb.init(project=log_id)

    wandb.run.name = log_name

    wandb.config.update(args)

    probe_train_loader, probe_valid_loader, _, classes = load_dataset(root=args.dataset_path, name=args.dataset_name, image_size=args.image_size,
                                                                    train_batch_size=32, valid_batch_size=32)
    worker_list = []
    split = [1.0 / args.size for _ in range(args.size)]
    temp_valid = 0
    temp_iteration = 0
    for rank in range(args.size):
        train_loader, _, _, classes = load_dataset(root=args.dataset_path, name=args.dataset_name, image_size=args.image_size, 
                                                    train_batch_size=args.batch_size, 
                                                    distribute=True, rank=rank, split=split, seed=args.seed)
        model = load_model(args.model, classes, pretrained=args.pretrained).to(args.device)
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)

        scheduler = Warmup_MultiStepLR(optimizer, warmup_step=args.warmup_step, milestones=args.milestones, gamma=args.gamma)
        # scheduler = Local_MultiStepLR(optimizer, warmup_step=args.warmup_step, gamma=args.gamma)

        if args.amp:
            worker = Worker_Vision_AMP(model, rank, optimizer, scheduler, train_loader, args.device)
        else:
            if args.optimization == "base":
                worker = Worker_Vision(model, rank, optimizer, scheduler, args.eta, args.rho, train_loader, args.device, False)
            else:
                worker = Worker_Vision(model, rank, optimizer, scheduler, args.eta, args.rho, train_loader, args.device, True)
        #worker_list.append(DDP(worker, device_ids=[gpu_id]]))
        worker_list.append(worker)


    # Main virutal global model initiate.
    center_model = copy.deepcopy(worker_list[0].model)

    center_model = CalculateCenter(center_model, worker_list, args.size)

    # Calculating Consensusdistance
    threshold = CalculateConsensus(center_model, worker_list, args.size)

    P = generate_P(args.mode, args.size) # generate gossip matrix
    iteration = 0
    for epoch in range(args.epoch):  
        for worker in worker_list:
            worker.update_iter()   
        for _ in range(train_loader.__len__()):
            if args.mode == 'csgd':
                for worker in worker_list:
                    worker.model.load_state_dict(center_model.state_dict())
                    worker.step()
                    worker.update_grad()
            else: # dsgd
                # 每个iteration，传播矩阵P中的worker做random shuffle（自己的邻居在下一个iteration时改变）
                if args.shuffle == "random":
                    P_perturbed = np.matmul(np.matmul(PermutationMatrix(args.size).T,P),PermutationMatrix(args.size)) 
                elif args.shuffle == "fixed":
                    P_perturbed = P
                model_dict_list = []
                
                
                # t+1/2 step
                for worker in worker_list:\
                    server = copy.deepcopy(worker.model.state_dict())
                    if (args.optimization == 'base'):
                        worker.step()
                    
                    elif (args.optimization == 'prox'):
                        worker.step_prox(server,args.mu)
    
                        # print("upgrde")
                    elif(args.optimization == 'sam'):
                        # gradient update (half update step)
                        worker.step_sam()

                    elif(args.optimization == 'samprox'):
                        worker.step_samprox2(server,args.mu)
                        
                    worker.update_grad() # -gradient
                    model_dict_list.append(worker.model.state_dict()) # update된 model weigth 들어감,
                
                #t+1 step
                if((iteration % args.local_iter == 0)):
                    for worker in worker_list:
                        for name, param in worker.model.named_parameters():
                            param.data = torch.zeros_like(param.data)
                            for i in range(args.size):
                                p = P_perturbed[worker.rank][i]
                                param.data += model_dict_list[i][name].data * p  # gossip algorithm.  
                                            

            # Main virutal global model update and calculating consensus distance
            center_model = CalculateCenter(center_model, worker_list, args.size)

            if iteration % 50 == 0:    
                start_time = datetime.datetime.now() 
                eval_iteration = iteration
                if args.amp:
                    train_acc, train_loss, valid_acc, valid_loss = eval_vision_amp(center_model, probe_train_loader, probe_valid_loader,
                                                                                None, iteration, wandb, args.device)                    
                else:
                    train_acc, train_loss, valid_acc, valid_loss = eval_vision(center_model, probe_train_loader, probe_valid_loader,
                                                                                None, iteration, wandb, args.device)
                # print(f"\n|\033[0;31m Iteration:{iteration}|{args.early_stop}, epoch: {epoch}|{args.epoch},\033[0m",
                        # f'train loss:{train_loss:.4}, acc:{train_acc:.4%}, '
                        # f'valid loss:{valid_loss:.4}, acc:{valid_acc:.4%}.',
                        # flush=True, end="\n")
                if temp_valid < valid_acc :
                    temp_valid = valid_acc
                    temp_iteration = iteration
            else:
                end_time = datetime.datetime.now()
                # print(f"\r|\033[0;31m Iteration:{eval_iteration}-{iteration}, time: {(end_time - start_time).seconds}s\033[0m", flush=True, end="")
            # iteration += 1
            # Calculate Current Consensus distance (threshold)        
            threshold = CalculateConsensus(center_model, worker_list, args.size)

            # multiple local gossip stage to make consensus.
            if (args.multiple_local_gossip) and (args.mode != 'csgd'):
                if((iteration % args.gossip_step == 0)):
                    current_distance = threshold
                    current_iteration = 0
                    while current_distance <  4: 
                        if current_iteration > 4:
                            break
                        model_dict_list_ = []
                        for worker in worker_list:
                            model_dict_list_.append(worker.model.state_dict()) 
                        current_iteration += 1
                        
                        for worker in worker_list:
                            for name, param in worker.model.named_parameters():
                                param.data = torch.zeros_like(param.data)
                                for i in range(args.size):
                                    p = P_perturbed[worker.rank][i]
                                    param.data += model_dict_list_[i][name].data * p
                                
                    
                    center_model = CalculateCenter(center_model, worker_list, args.size)
                    current_distance = CalculateConsensus(center_model, worker_list, args.size)
                    # writer.add_scalar(f"Consensus distance in iteration {iteration}", current_distance, threshold)
                    # wandb.log({\
                    #     'Consensus distance': current_distance,
                    #     'threshold' : threshold
                    # })
                        # print(f"\niteration:{iteration}, threshold:{threshold},  current_distance:{current_distance}, current iteration: {current_iteration}" )
            iteration += 1
            if iteration == args.early_stop: 
                start_time = datetime.datetime.now() 
                eval_iteration = iteration
                if args.amp:
                    train_acc, train_loss, valid_acc, valid_loss = eval_vision_amp(center_model, probe_train_loader, probe_valid_loader,
                                                                                None, iteration, wandb, args.device)                    
                else:
                    train_acc, train_loss, valid_acc, valid_loss = eval_vision(center_model, probe_train_loader, probe_valid_loader,
                                                                                None, iteration, wandb, args.device)
                # print(f"\n|\033[0;31m Iteration:{iteration}|{args.early_stop}, epoch: {epoch}|{args.epoch},\033[0m",
                #         f'train loss:{train_loss:.4}, acc:{train_acc:.4%}, '
                #         f'valid loss:{valid_loss:.4}, acc:{valid_acc:.4%}.',
                #         flush=True, end="\n")
                if temp_valid < valid_acc :
                    temp_valid = valid_acc
                    temp_iteration = iteration
                    
                print(f"\n|\033[0;31m Best Iteration:{temp_iteration}|",
                        f'Best acc:{temp_valid:.4%}.',
                        flush=True, end="\n")
                break
        if iteration == args.early_stop: break

        
        

        

    state = {
        'acc': train_acc,
        'epoch': epoch,
        'state_dict': center_model.state_dict() 
    }    
    if not os.path.exists(args.perf_dict_dir):
        os.mkdir(args.perf_dict_dir)  
    torch.save(state, os.path.join(args.perf_dict_dir, log_id + '.t7'))
     
    print('ending')

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    ## dataset
    parser.add_argument("--dataset_path", type=str, default='datasets')
    parser.add_argument("--dataset_name", type=str, default='CIFAR10',
                                            choices=['CIFAR10','CIFAR100','TinyImageNet'])
    parser.add_argument("--image_size", type=int, default=32, help='input image size')
    parser.add_argument("--batch_size", type=int, default=32)
    
    # mode parameter
    parser.add_argument('--mode', type=str, default='csgd', choices=['csgd', 'ring', 'meshgrid', 'exponential','all'])
    parser.add_argument('--shuffle', type=str, default="fixed", choices=['fixed', 'random'])
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--port', type=int, default=29500)

    # deep model parameter
    parser.add_argument('--model', type=str, default='ResNet18', 
                        choices=['ResNet18', 'AlexNet', 'DenseNet121', 'ResNet34'])
    parser.add_argument("--pretrained", type=int, default=0)

    #optimization
    parser.add_argument("--optimization", type=str, default='base', choices=['base', 'prox','sam', 'samprox'])

    #SAM parameter
    parser.add_argument('--eta', type=float, default=0.01, help='eta value')
    parser.add_argument('--rho', type=float, default=0.5,  help='rho value') 

    parser.add_argument('--mu', type=float, default=1.0,  help='mu value') 
    

    # multiple gossip stage  
    parser.add_argument("--multiple_local_gossip", action='store_true', help='Run multiple gossip algorithm')

    #local iteration 
    parser.add_argument("--local_iter", type=int, default=1)

    # optimization parameter
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0005,  help='weight decay')
    parser.add_argument('--gamma', type=float, default=0.998)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--warmup_step', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--early_stop', type=int, default=5000, help='w.r.t., iterations')
    parser.add_argument('--milestones', type=int, nargs='+', default=[1000, 1500, 2000, 2500])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--amp", action='store_true', help='automatic mixed precision')

    # concensus distance parameter
    parser.add_argument("--gossip_step", type=int, default=300)
    # parser.add_argument("--only_dec", action='store_true', help='only control consensus distance in decay pahse')

    parser.add_argument("--exp_name", type=str, default='Decentralized')


    # test version
    parser.add_argument("--using_efficient_scheme", action='store_true', help='using efficient Consensus control scheme')
    args = parser.parse_args()

    args = add_identity(args, dir_path)
    # print(args)
    main(args)
