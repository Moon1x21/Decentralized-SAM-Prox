import torch 
import numpy as np
from collections import defaultdict

class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01, scheduling='base'):
        self.optimizer = optimizer
        self.model = model
        self.eta = eta
        self.state = defaultdict(dict)
        self.scheduler = scheduling
        if(self.scheduler != 'base'):
            if(self.scheduler == 'log'):
                self.rho_list = np.round(np.logspace(-1.3,-0.3, 10),3)
            elif(self.scheduler == 'step'):
                self.rho_list = np.clip(np.array([rho * round(pow(0.1,(1-int(i / (5000 / 2)))),2) for i in range(5000)]),0,1)
            self.index = 0
            self.rho = self.rho_list[0]
        else:
            self.rho = rho
               
    @torch.no_grad()
    def ascent_step(self):
        # print('defaulstate:', self.state.keys())
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1e-12
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)

        #update rho
        if(self.scheduler != 'base'):
            self.index += 1
            self.rho = self.rho_list[self.index]

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        
        # self.optimizer.step()

        
        
class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1e-12
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
