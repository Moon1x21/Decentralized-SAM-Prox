
import torch
import torch.nn as nn
import copy
from utils.minimizers import *


criterion = nn.CrossEntropyLoss()

class SAMProxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, preds, new_weight=None, central_weight=None, mu=1.0):
        weight_diff = sum((torch.pow((x - y), 2).sum() for x, y in zip(
            new_weight.state_dict().values(), central_weight.values())))
        return criterion(outputs, preds) + mu / 2 * weight_diff

smaproxloss = SAMProxLoss()

class Worker_Vision:
    def __init__(self, model, rank, optimizer, scheduler, eta, rho,
                 train_loader, device, optimization='base', scheduling='base'):       
        self.model = model
        self.rank = rank
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.train_loader_iter = train_loader.__iter__()
        self.device = device
        self.localprox = model.state_dict()
        self.optimization = optimization
        # import pdb; pdb.set_trace()
        if(self.optimization in ['sam','samprox','samAdmm']):
            self.rho = rho
            self.eta = eta
            self.sam_optimizer = SAM(self.optimizer, self.model, self.rho, self.eta, scheduling)
        
            if(self.optimization == 'samAdmm'):
                temp = copy.deepcopy(self.model)
                for p in temp.parameters():
                    p = 0
                self.dual_y = copy.deepcopy(temp.state_dict())
                self.dual_z = copy.deepcopy(temp.state_dict())
       
        
    def update_iter(self):
        self.train_loader_iter = self.train_loader.__iter__()

    def step(self):
        self.model.train()

        batch = next(self.train_loader_iter)
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        # self.optimizer.step()
    
    def step_prox(self,mu):
        self.model.train()

        batch = next(self.train_loader_iter)
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = criterion(output, target)
        self.optimizer.zero_grad()
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.grad.add_(mu*(p.data-self.localprox[n].data))
         
        loss.backward()
        # self.optimizer.step()

    def step_sam(self):
        self.model.train()

        batch = next(self.train_loader_iter)
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward() # gradient 구함.
        self.sam_optimizer.ascent_step()
        
        
        self.optimizer.zero_grad()
        criterion(self.model(data),target).backward()
        self.sam_optimizer.descent_step()
                    
    def step_samprox(self,mu):
        self.model.train()
        
        batch = next(self.train_loader_iter)
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward() # gradient 구함.
        self.sam_optimizer.ascent_step()
        
        
        self.optimizer.zero_grad()
        criterion(self.model(data),target).backward() # gradient

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.grad.add_(mu*(p.data-self.localprox[n].data))
         
        self.sam_optimizer.descent_step()

    def step_samprox_scheduling(self,mu):
        self.model.train()

        batch = next(self.train_loader_iter)
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward() # gradient 구함.
        self.sam_optimizer.update_rho()
        self.sam_optimizer.ascent_step()


        self.optimizer.zero_grad()
        criterion(self.model(data),target).backward() # gradient

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.grad.add_(mu*(p.data-self.localprox[n].data))
            
        self.sam_optimizer.descent_step()
    
    def step_samADMM(self,mu):
        self.model.train()
        
        batch = next(self.train_loader_iter)
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward() # gradient 구함.
        self.sam_optimizer.ascent_step()
        
        
        self.optimizer.zero_grad()
        criterion(self.model(data),target).backward() # gradient

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.grad.add_(mu*(p.data-self.localprox[n].data) + self.dual_y[n]*(p.data))
            
        self.sam_optimizer.descent_step()

        self.update_dualy(mu)


    def step_csgd(self):
        self.model.train()

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()


        grad_dict = {}
        for name, param in self.model.named_parameters():
            grad_dict[name] = param.grad.data

        return grad_dict

    def update_grad(self):
        self.optimizer.step()
        self.scheduler.step()

    def scheduler_step(self):
        self.scheduler.step()

    def update_dualy(self,mu): 
        for n, p in self.model.named_parameters():
            self.dual_y[n] += mu * (p.data-self.dual_y[n].data)

    # def update_dualz(self,mu,worker_list,W):
    #     for n,p in self.mdoel.named_parameters():
    #         for worker in range(worker_list):
    #             w = P_perturbed[worker.rank][i]
    #             self.dual_z[n] += mu * w * ( self.localprox[n].data - worker.localprox[n].data)

from torch.cuda.amp.grad_scaler import GradScaler
scaler = GradScaler()
class Worker_Vision_AMP:
    def __init__(self, model, rank, optimizer, scheduler,
                 train_loader, device):       
        self.model = model
        self.rank = rank
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        # self.train_loader_iter = train_loader.__iter__()
        self.device = device


    def update_iter(self):
        self.train_loader_iter = self.train_loader.__iter__()

    def step(self):
        self.model.train()

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        with torch.cuda.amp.autocast(enabled=True,dtype=torch.float16):
            output = self.model(data)
            loss = criterion(output, target)
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()

    def step_csgd(self):
        self.model.train()

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        with torch.cuda.amp.autocast(enabled=True,dtype=torch.float16):
            output = self.model(data)
            loss = criterion(output, target)
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()

        grad_dict = {}
        for name, param in self.model.named_parameters():
            grad_dict[name] = param.grad.data

        return grad_dict

    def update_grad(self):
        # self.optimizer.step()
        scaler.step(self.optimizer)
        scaler.update()
        self.scheduler.step()

    def scheduler_step(self):
        self.scheduler.step()
