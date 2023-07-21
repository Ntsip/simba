import torch
from torch.optim.optimizer import Optimizer
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Simba(Optimizer):
    """
    This is a pytorch implementation of "Simba: A Scalable Bilevel Preconditioned Gradient Method for Fast Evasion of Flat Areas and Saddle Points"

    Arguments:
        params (iterable) -- iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) -- learning rate (default: 0.01)
        momentum (float, optional) -- coefficient used for computing running averages of gradient (default: 0.9)
        coarse_dim_perc (float, optional) -- number of coarse model dimensions in percentage of the fine model dimensions (default: 0.5)
        rank (int, optional) -- number of eigevalues and eigenvectors to be computed by the T-SVD
        eps (float, optional) -- lower bound on eigenvalues to improve numerical stability (default: 1e-8)
        weight_decay (float, optional) -- weight decay (L2 penalty) (default: 0.0)
    """

    def __init__(self, params, lr=1e-2, momentum=0.9, coarse_dim_perc=0.5, rank=20,  eps=1e-8, weight_decay=0.):

        defaults = dict(lr=lr, momentum=momentum, coarse_dim_perc=coarse_dim_perc, rank=rank, weight_decay=weight_decay,
                        eps=eps)
        super(Simba, self).__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            grad_avgs = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p.data)
                    grads.append(p.grad.data)
                    state = self.state[p]
                    if len(state) == 0:
                        state['grad_avgs'] = torch.zeros_like(p.data)
                        state['step'] = torch.tensor(0.)
                    grad_avgs.append(state['grad_avgs'])

            simba(params_with_grad,
                grads,
                grad_avgs,
                momentum=group['momentum'],
                coarse_dim_perc=group['coarse_dim_perc'],
                rank=group['rank'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                lr=group['lr']
                )
        return loss

def simba(params, grads, grad_avgs, momentum, coarse_dim_perc, rank, weight_decay, eps, lr):

    for i, param in enumerate(params):
        grad = grads[i]
        grad_avg = grad_avgs[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
    
        grad_avg.mul_(momentum).add_(grad, alpha=1)

        n_weights = grad.shape[0]

        coarse_dim = math.ceil(coarse_dim_perc * n_weights) + 1

        idx = torch.randperm(n_weights)[:coarse_dim]

        grad_avg_mat = grad_avg.view(param.shape[0], -1)
        grad_avg_sub = grad_avg_mat[idx]

        precond_reduced = grad_avg_sub @ grad_avg_sub.t()

        if rank < precond_reduced.shape[0]:
            U_r, Sigma_r, v = torch.svd_lowrank(precond_reduced, q=rank)
        else:
            U_r, Sigma_r, v = torch.linalg.svd(precond_reduced)

        Sigma_r = Sigma_r.abs().sqrt()

        if eps != 0:
            Sigma_r[Sigma_r < eps] = eps

        U_r_minus1 = U_r[:, torch.arange(U_r.size(1)) != U_r.size(1) - 1]
        Sigma_minus1_inv = torch.diag(
            Sigma_r[torch.arange(Sigma_r.size(0)) != Sigma_r.size(0) - 1] ** (-1) - Sigma_r[
                -1] ** (-1))

        a_term = U_r_minus1.t() @ grad_avg_sub
        b_term = Sigma_minus1_inv @ a_term
        c_term = U_r_minus1 @ b_term
        dH = - grad_avg_sub / Sigma_r[-1] - c_term

        d = torch.zeros_like(grad_avg_mat)
        d[idx] = dH

        d_unflat = d.view(param.shape)

        param.add_(d_unflat, alpha=lr)
