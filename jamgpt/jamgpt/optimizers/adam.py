import torch
import torch.distributed as dist
from torch import Tensor


class DistAdamW(torch.optim.Optimizer):
    def __init__(
        self, param_groups, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(DistAdamW, self).__init__(param_groups, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(
                    dist.reduce_scatter_tensor(
                        grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                    ).get_future()
                )
                grad_slices.append(grad_slice)
        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            params: list[Tensor] = group["params"]
            eps = group["eps"]
            wd = group["weight_decay"]
            lr = group["lr"] * getattr(p, "lr_mul", 1.0)
            for base_i in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base_i]
                rank_size = p.shape[0] // world_size
                grad_slice = grad_slices[idx]
                state = self.state[p]
                p_slice = p[rank * rank_size : (rank + 1) * rank_size]
                if not state:
                    state["step"] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p_slice)
                    state["exp_avg_sq"] = torch.zeros_like(p_slice)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                if wd != 0:
                    p_slice.mul_(1 - lr * wd * getattr(p, "wd_mul", 1.0))
                exp_avg.mul_(beta1).add_(grad_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_slice, grad_slice, alpha=1 - beta2)
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias_correction2) / bias_correction1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1)
                idx += 1
                all_reduce_futures.append(
                    dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                )
        torch.futures.wait_all(all_reduce_futures)
