
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
                 train_loader, device, sam=False):       
        self.model = model
        self.rank = rank
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        # self.train_loader_iter = train_loader.__iter__()
        self.device = device
        if(sam):
            self.rho = rho
            self.eta = eta
            self.sam_optimizer = SAM(self.optimizer, self.model, self.rho, self.eta)
            

        
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


    def step_samprox(self,server,mu):
        self.model.train()
        
        batch = next(self.train_loader_iter)
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward() # gradient 구함.
        self.sam_optimizer.ascent_step()
        
        
        self.optimizer.zero_grad()
        smaproxloss(self.model(data),target, self.model,server,mu).backward() # gradient 
        self.sam_optimizer.descent_step()

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
