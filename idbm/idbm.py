import os

import fire
import numpy as np
import torch as th
import wandb
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from unet.unet import UNetModel

data_path = os.path.expanduser("~/torch-data/")
device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")

IDBM = "IDBM"
DIPF = "DIPF"

# Routines -----------------------------------------------------------------------------


# Reference SDE with law R: $dx_t = sigma dw_t$, $σ ≥ 0, t ∈ [0, 1].$
# Shapes: x_t is [B × C × H x W], x is [T, B × C × H x W].
# Fwd and bwd inferred SDEs formulated on increasing timescales.
# Implementation allows σ = 0.


def sample_bridge(x_0, x_1, t, sigma):
    t = t[:, None, None, None]
    mean_t = (1.0 - t) * x_0 + t * x_1
    var_t = sigma**2 * t * (1.0 - t)
    z_t = th.randn_like(x_0)
    x_t = mean_t + th.sqrt(var_t) * z_t
    return x_t


def idbm_target(x_t, x_1, t):
    target_t = (x_1 - x_t) / (1.0 - t[:, None, None, None])
    return target_t


def drift_target(x_t, x_t_dt, dt):
    dt = th.full(size=[x_t.shape[1], 1, 1, 1], fill_value=dt, device=device)
    target_t = (x_t_dt - x_t) / dt[:, None, None, None]
    return target_t


def euler_discretization(x, xp, nn, sigma):
    # Assumes x has shape [T, B, C, H, W].
    # Assumes x[0] already initialized.
    # We normalize by D = C * H * W the drift squared norm, and not by scalar sigma.
    # Fills x[1] to x[T] and xp[0] to xp[T - 1].
    T = x.shape[0] - 1  # Discretization steps.
    B = x.shape[1]
    dt = th.full(size=(x.shape[1],), fill_value=1.0 / T, device=device)
    drift_norms = 0.0
    for i in range(1, T + 1):
        t = dt * (i - 1)
        alpha_t = nn(x[i - 1], None, t)
        drift_norms = drift_norms + th.mean(alpha_t.view(B, -1) ** 2, dim=1)
        xp[i - 1] = x[i - 1] + alpha_t * (1 - t[:, None, None, None])
        drift_t = alpha_t * dt[:, None, None, None]
        eps_t = th.randn_like(x[i - 1])
        diffusion_t = sigma * th.sqrt(dt[:, None, None, None]) * eps_t
        x[i] = x[i - 1] + drift_t + diffusion_t
    drift_norms = drift_norms / T
    return drift_norms.cpu()


# Data ---------------------------------------------------------------------------------


class EMNIST(datasets.EMNIST):
    def __init__(self, **kwargs):
        super().__init__(split="letters", **kwargs)
        indices = (self.targets <= 5).nonzero(as_tuple=True)[0]
        self.data, self.targets = (
            self.data[indices].transpose(1, 2),
            self.targets[indices],
        )


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.epoch_iterator)
        except StopIteration:
            self.epoch_iterator = super().__iter__()
            batch = next(self.epoch_iterator)
        return batch


def train_iter(data, batch_dim):
    return iter(
        InfiniteDataLoader(
            dataset=data,
            batch_size=batch_dim,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
    )


def test_loader(data, batch_dim):
    return DataLoader(
        dataset=data,
        batch_size=batch_dim,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )


def resample_indices(from_n, to_n):
    # Equi spaced resampling, first and last element always included.
    return np.round(np.linspace(0, from_n - 1, num=to_n)).astype(int)


def image_grid(x, normalize=False, n=5):
    img = x[: n**2].cpu()
    img = make_grid(img, nrow=n, normalize=normalize, scale_each=normalize)
    img = wandb.Image(img)
    return img


# For fixed permutations of test sets:
rng = np.random.default_rng(seed=0x87351080E25CB0FAD77A44A3BE03B491)

# Linear scaling to float [-1.0, 1.0]:
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
tr_data_0 = datasets.MNIST(
    root=data_path + "mnist", train=True, download=True, transform=transform
)
te_data_0 = datasets.MNIST(
    root=data_path + "mnist", train=False, download=True, transform=transform
)
te_data_0 = Subset(te_data_0, rng.permutation(len(te_data_0)))
tr_data_1 = EMNIST(
    root=data_path + "emnist", train=True, download=True, transform=transform
)
te_data_1 = EMNIST(
    root=data_path + "emnist", train=False, download=True, transform=transform
)
te_data_1 = Subset(te_data_1, rng.permutation(len(te_data_1)))


# NN Model -----------------------------------------------------------------------------


def init_nn():
    # From https://github.com/openai/guided-diffusion/tree/main/guided_diffusion:
    return UNetModel(
        in_channels=1,
        model_channels=128,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(),
        dropout=0.1,
        channel_mult=(0.5, 1, 1),
        num_heads=4,
        use_scale_shift_norm=True,
        temb_scale=1000,
    )


class EMAHelper:
    # Simplified from https://github.com/ermongroup/ddim/blob/main/models/ema.py:
    def __init__(self, module, mu=0.999, device=None):
        self.module = module
        self.mu = mu
        self.device = device
        self.shadow = {}
        # Register:
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self):
        locs = self.module.locals
        module_copy = type(self.module)(*locs).to(self.device)
        module_copy.load_state_dict(self.module.state_dict())
        self.ema(module_copy)
        return module_copy


# Run ----------------------------------------------------------------------------------


def run(
    method=IDBM,
    sigma=1.0,
    iterations=60,
    training_steps=5000,
    discretization_steps=30,
    batch_dim=128,
    learning_rate=1e-4,
    grad_max_norm=1.0,
    ema_decay=0.999,
    cache_steps=250,
    cache_batch_dim=2560,
    test_steps=5000,
    test_batch_dim=500,
    loss_log_steps=100,
    imge_log_steps=1000,
):
    config = locals()
    assert isinstance(sigma, float) and sigma >= 0
    assert isinstance(learning_rate, float) and learning_rate > 0
    assert isinstance(grad_max_norm, float) and grad_max_norm >= 0
    assert method in [IDBM, DIPF]

    console = Console(log_path=False)
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        TextColumn("•"),
        MofNCompleteColumn(),
        console=console,
        speed_estimate_period=60 * 5,
    )
    iteration_t = progress.add_task("iteration", total=iterations)
    step_t = progress.add_task("step", total=iterations * training_steps)

    wandb.init(project="idbm-x", config=config)
    console.log(wandb.config)

    tr_iter_0 = train_iter(tr_data_0, batch_dim)
    tr_iter_1 = train_iter(tr_data_1, batch_dim)
    tr_cache_iter_0 = train_iter(tr_data_0, cache_batch_dim)
    tr_cache_iter_1 = train_iter(tr_data_1, cache_batch_dim)
    te_loader_0 = test_loader(te_data_0, test_batch_dim)
    te_loader_1 = test_loader(te_data_1, test_batch_dim)

    bwd_nn = init_nn().to(device)
    fwd_nn = init_nn().to(device)

    bwd_ema = EMAHelper(bwd_nn, ema_decay, device)
    fwd_ema = EMAHelper(fwd_nn, ema_decay, device)
    bwd_sample_nn = bwd_ema.ema_copy()
    fwd_sample_nn = fwd_ema.ema_copy()

    bwd_nn.train()
    fwd_nn.train()
    bwd_sample_nn.eval()
    fwd_sample_nn.eval()

    bwd_optim = th.optim.Adam(bwd_nn.parameters(), lr=learning_rate)
    fwd_optim = th.optim.Adam(fwd_nn.parameters(), lr=learning_rate)

    dt = 1.0 / discretization_steps
    t_T = 1.0 - dt * 0.5

    s_path = th.zeros(
        size=(discretization_steps + 1,) + (cache_batch_dim, 1, 28, 28), device=device
    )  # i: 0, ..., discretization_steps;     t: 0, dt, ..., 1.0.
    p_path = th.zeros(
        size=(discretization_steps,) + (cache_batch_dim, 1, 28, 28), device=device
    )  # i: 0, ..., discretization_steps - 1; t: 0, dt, ..., 1.0 - dt.

    progress.start()
    step = 0
    for iteration in range(1, iterations + 1):
        console.log(f"iteration {iteration}: {step}")
        progress.update(iteration_t, completed=iteration)
        # Setup:
        if (iteration % 2) != 0:
            # Odd iteration => bwd.
            direction = "bwd"
            nn = bwd_nn
            ema = bwd_ema
            sample_nn = bwd_sample_nn
            optim = bwd_optim
            te_loader_x_0 = te_loader_1
            te_loader_x_1 = te_loader_0

            def sample_idbm_coupling(step):
                if iteration == 1:
                    # Independent coupling:
                    x_0 = next(tr_iter_1)[0].to(device)
                    x_1 = next(tr_iter_0)[0].to(device)
                else:
                    with th.no_grad():
                        if (step - 1) % cache_steps == 0:
                            console.log(f"cache update: {step}")
                            # Simulate previously inferred SDE:
                            s_path[0] = next(tr_cache_iter_0)[0].to(device)
                            euler_discretization(s_path, p_path, fwd_sample_nn, sigma)
                        # Random selection:
                        idx = th.randperm(cache_batch_dim, device=device)[:batch_dim]
                        # Reverse path:
                        x_0, x_1 = s_path[-1, idx], s_path[0, idx]
                return x_0, x_1

            def sample_dipf_path(step):
                with th.no_grad():
                    if (step - 1) % cache_steps == 0:
                        console.log(f"cache update: {step}")
                        # Simulate previously inferred SDE:
                        # NN initialized at 0.0 => first iteration == refence SDE.
                        s_path[0] = next(tr_cache_iter_0)[0].to(device)
                        euler_discretization(s_path, p_path, fwd_sample_nn, sigma)
                    # Random selection:
                    idx = th.randperm(cache_batch_dim, device=device)[:batch_dim]
                    # Reverse path:
                    x_path = th.flip(s_path[:, idx], [0])
                return x_path

        else:
            # Even iteration => fwd.
            direction = "fwd"
            nn = fwd_nn
            ema = fwd_ema
            sample_nn = fwd_sample_nn
            optim = fwd_optim
            te_loader_x_0 = te_loader_0
            te_loader_x_1 = te_loader_1

            def sample_idbm_coupling(step):
                with th.no_grad():
                    if (step - 1) % cache_steps == 0:
                        console.log(f"cache update: {step}")
                        # Simulate previously inferred SDE:
                        s_path[0] = next(tr_cache_iter_1)[0].to(device)
                        euler_discretization(s_path, p_path, bwd_sample_nn, sigma)
                    # Random selection:
                    idx = th.randperm(cache_batch_dim, device=device)[:batch_dim]
                    # Reverse path:
                    x_0, x_1 = s_path[-1, idx], s_path[0, idx]
                return x_0, x_1

            def sample_dipf_path(step):
                with th.no_grad():
                    if (step - 1) % cache_steps == 0:
                        console.log(f"cache update: {step}")
                        # Simulate previously inferred SDE:
                        s_path[0] = next(tr_cache_iter_1)[0].to(device)
                        euler_discretization(s_path, p_path, bwd_sample_nn, sigma)
                    # Random selection:
                    idx = th.randperm(cache_batch_dim, device=device)[:batch_dim]
                    # Reverse path:
                    x_path = th.flip(s_path[:, idx], [0])
                return x_path

        for step in range(step + 1, step + training_steps + 1):
            progress.update(step_t, completed=step)
            optim.zero_grad()

            if method == IDBM:
                x_0, x_1 = sample_idbm_coupling(step)
                t = th.rand(size=(batch_dim,), device=device) * t_T
                x_t = sample_bridge(x_0, x_1, t, sigma)
                target_t = idbm_target(x_t, x_1, t)
            elif method == DIPF:
                x_path = sample_dipf_path(step)
                t_i = th.randint(
                    0, discretization_steps, size=(batch_dim,), device=device
                )
                t = t_i.to(th.float32) * dt
                x_t = th.stack([x_path[ti, i] for i, ti in enumerate(t_i)])
                x_t_dt = th.stack([x_path[ti, i] for i, ti in enumerate(t_i + 1)])
                target_t = drift_target(x_t, x_t_dt, dt)

            alpha_t = nn(x_t, None, t)
            losses = (target_t - alpha_t) ** 2
            losses = th.mean(losses.view(losses.shape[0], -1), dim=1)
            if method == DIPF:
                losses = losses / sigma**2
            loss = th.mean(losses)

            loss.backward()
            if grad_max_norm > 0:
                grad_norm = th.nn.utils.clip_grad_norm_(nn.parameters(), grad_max_norm)
            optim.step()

            ema.update()

            if step % test_steps == 0:
                console.log(f"test: {step}")
                ema.ema(sample_nn)
                with th.no_grad():
                    te_s_path = th.zeros(
                        size=(discretization_steps + 1,) + (test_batch_dim, 1, 28, 28),
                        device=device,
                    )
                    te_p_path = th.zeros(
                        size=(discretization_steps,) + (test_batch_dim, 1, 28, 28),
                        device=device,
                    )
                    # Assumes data is in [0.0, 1.0], scale appropriately:
                    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
                    drift_norm = []
                    for te_x_0, te_x_1 in zip(te_loader_x_0, te_loader_x_1):
                        te_x_0, te_x_1 = te_x_0[0].to(device), te_x_1[0].to(device)
                        te_x_1 = (te_x_1 + 1.0) / 2.0
                        te_s_path[0] = te_x_0
                        drift_norm.append(
                            euler_discretization(te_s_path, te_p_path, sample_nn, sigma)
                        )
                        te_s_path = th.clip((te_s_path + 1.0) / 2.0, 0.0, 1.0)
                        te_p_path = th.clip((te_p_path + 1.0) / 2.0, 0.0, 1.0)
                        fid_metric.update(te_x_1.expand(-1, 3, -1, -1), real=True)
                        if method == IDBM:
                            fid_idx = -2
                        elif method == DIPF:
                            fid_idx = -1
                        fid_metric.update(
                            te_p_path[fid_idx].expand(-1, 3, -1, -1), real=False
                        )
                    drift_norm = th.mean(th.cat(drift_norm)).item()
                    fid = fid_metric.compute().item()
                    wandb.log({f"{direction}/test/drift_norm": drift_norm}, step=step)
                    wandb.log({f"{direction}/test/fid": fid}, step=step)
                    for i, ti in enumerate(
                        resample_indices(discretization_steps + 1, 5)
                    ):
                        wandb.log(
                            {f"{direction}/test/x[{i}-{5}]": image_grid(te_s_path[ti])},
                            step=step,
                        )
                    for i, ti in enumerate(resample_indices(discretization_steps, 5)):
                        wandb.log(
                            {f"{direction}/test/p[{i}-{5}]": image_grid(te_p_path[ti])},
                            step=step,
                        )

            if step % loss_log_steps == 0:
                wandb.log({f"{direction}/train/loss": loss.item()}, step=step)
                wandb.log({f"{direction}/train/grad_norm": grad_norm}, step=step)

            if step % imge_log_steps == 0:
                if method == DIPF:
                    x_0 = x_path[0]
                    x_1 = x_path[-1]
                wandb.log({f"{direction}/train/x_0": image_grid(x_0, True)}, step=step)
                wandb.log({f"{direction}/train/x_1": image_grid(x_1, True)}, step=step)

            if step % training_steps == 0:
                console.log(f"EMA update: {step}")
                # Make sure EMA is updated at the end of each iteration:
                ema.ema(sample_nn)
    progress.stop()


if __name__ == "__main__":
    fire.Fire(run)
