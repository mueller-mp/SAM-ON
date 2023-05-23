import torch
from collections import defaultdict

class ASAM_BN:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01, adaptive=True, p='2', normalize_bias=False, elementwise=True, layerwise=False, no_bn=False, only_bn=False):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
        self.adaptive=adaptive
        self.p=p
        self.normalize_bias=normalize_bias
        self.elementwise=elementwise
        self.layerwise=layerwise
        self.only_bn=only_bn
        self.no_bn=no_bn
        assert not (self.only_bn and self.no_bn)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            self.state[p]['old_p'] = p.data.clone()
            self.state[p]['old_p'] = p.data.clone()
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                self.state[p]['perturbed']=False
                continue
            self.state[p]['perturbed']=True
            t_w = self.state[p].get("eps")
            if t_w is None: # initialize t_w
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if ('weight' in n) or self.normalize_bias: # compute t_w and modify grad to t_w*grad
                t_w[...] = p[...]
                if self.elementwise:
                    t_w.abs_().add_(self.eta)
                elif self.layerwise:
                    t_w.data = torch.norm(p.data)*torch.ones_like(p.data).add_(self.eta).data
                if self.p == '2':
                    p.grad.mul_(t_w)  # update gradient
            if self.p == 'infinity':
                if ('weight' in n) or self.normalize_bias:
                    p.grad.sign_().mul_(t_w)
                else:
                    p.grad.sign_()
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16

        for n, p in self.model.named_parameters():
            if (p.grad is None) or (not self.state[p]['perturbed']):
                # p.requires_grad=True  # for runtime measurement
                continue
            t_w = self.state[p].get("eps") # get normalization operator
            if self.p=='2':
                if ('weight' in n) or self.normalize_bias: # second multiplication with t_w
                    p.grad.mul_(t_w)
                eps = t_w
                eps[...] = p.grad[...]
                eps.mul_(self.rho / wgrad_norm)
            elif self.p=='infinity':
                eps = t_w
                eps[...] = p.grad[...]*self.rho
            else:
                raise NotImplementedError
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if (p.grad is None) or (not self.state[p]['perturbed']):
                # p.requires_grad=False # for runtime measurement
                continue
            p.data = self.state[p]['old_p']

        self.optimizer.step()
        self.optimizer.zero_grad()

class SAM_BN(ASAM_BN):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            self.state[p]['old_p'] = p.data.clone()
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                self.state[p]['perturbed']=False
                continue
            self.state[p]['perturbed']=True
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if (p.grad is None) or (not self.state[p]['perturbed']):
                # p.requires_grad=True # for runtime measurement
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()


