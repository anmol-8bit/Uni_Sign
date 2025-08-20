# utils.py
"""
This file is modified from:
https://github.com/facebookresearch/deit/blob/main/utils.py
"""

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references, adjusted for DeepSpeed.
"""

import io
import os
import time, random
import numpy as np
from collections import defaultdict, deque
import datetime
import pickle
import gzip
import argparse

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# DeepSpeed / distributed
import deepspeed
from deepspeed.accelerator import get_accelerator
import torch.distributed as tdist
from typing import Optional, Dict, Any


# -----------------------------
# Distributed helpers (torch.distributed)
# -----------------------------

def is_dist_avail_and_initialized():
    return tdist.is_available() and tdist.is_initialized()

def get_world_size():
    return tdist.get_world_size() if is_dist_avail_and_initialized() else 1

def get_rank():
    return tdist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process():
    return get_rank() == 0


# -----------------------------
# W&B helpers (safe no-op when disabled)
# -----------------------------
def init_wandb(args, config: Optional[Dict[str, Any]] = None):
    """
    Create a wandb run on rank-0 only. Returns run or None.
    """
    if not getattr(args, "wandb", False) or not is_main_process():
        return None
    try:
        import wandb
        # Safe default if args.wandb_mode is missing/None
        mode = getattr(args, "wandb_mode", None) or os.environ.get("WANDB_MODE", "online")

        base_config = {
            "batch_size": args.batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "epochs": args.epochs,
            "lr": args.lr,
            "warmup_epochs": args.warmup_epochs,
            "dataset": args.dataset,
            "dtype": args.dtype,
            "zero_stage": args.zero_stage,
            "max_length": args.max_length,
            "world_size": get_world_size(),
        }
        if isinstance(config, dict):
            base_config.update(config)

        run = wandb.init(
            project=getattr(args, "wandb_project", "unisign"),
            entity=getattr(args, "wandb_entity", None),
            name=getattr(args, "wandb_run_name", None),
            group=getattr(args, "wandb_group", None),
            tags=getattr(args, "wandb_tags", None),
            id=getattr(args, "wandb_id", None),
            resume="allow" if getattr(args, "wandb_id", None) else None,
            mode=mode,
            config=base_config,
        )
        return run
    except Exception as e:
        print(f"[wandb] init disabled or failed: {e}")
        return None

def wandb_log(run, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
    """Safe logging; no-ops if run is None."""
    if run is None:
        return
    try:
        if step is not None:
            run.log(metrics, step=step, commit=commit)
        else:
            run.log(metrics, commit=commit)
    except Exception as e:
        print(f"[wandb] log failed: {e}")

def wandb_close(run):
    try:
        if run is not None:
            run.finish()
    except Exception:
        pass

def wandb_watch(run, model, log: str = "gradients", log_freq: int = 250):
    if run is None or not is_main_process():
        return
    try:
        import wandb
        wandb.watch(model, log=log, log_freq=log_freq)
    except Exception:
        pass


# -----------------------------
# Logging / metrics
# -----------------------------

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average. Safe on empty sequences.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        tdist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        if not self.deque:
            return float('nan')
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        if not self.deque:
            return float('nan')
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else float('nan')

    @property
    def max(self):
        return max(self.deque) if self.deque else float('nan')

    @property
    def value(self):
        return self.deque[-1] if self.deque else float('nan')

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if (i > 0 and i % print_freq == 0) or (i == len(iterable) - 1):
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds))) if eta_seconds == eta_seconds else "N/A"
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / max(1, len(iterable))))


# -----------------------------
# Misc helpers
# -----------------------------

def save_on_master(*args, **kwargs):
    """
    Torch save that only runs on rank-0 (main) to avoid multi-rank write clashes.
    Usage: save_on_master({'model': state_dict}, path)
    """
    if is_main_process():
        print("save ckpt begin")
        torch.save(*args, **kwargs)
        print("save ckpt finish")

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for _, v in model.named_parameters()) / 1e6

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """Workaround for ModelEma._load_checkpoint to accept an already-loaded object"""
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)

def setup_for_distributed(is_master_flag):
    """Disable printing when not in master process"""
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master_flag or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print


def init_distributed_mode(args):
    """Torch DDP init (not used when using DeepSpeed-only)"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    tdist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                             world_size=args.world_size, rank=args.rank)
    tdist.barrier()
    setup_for_distributed(args.rank == 0)

def init_distributed_mode_ds(args):
    """DeepSpeed distributed init"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    deepspeed.init_distributed()
    tdist.barrier()
    setup_for_distributed(args.rank == 0)

def sampler_func(clip, sn, random_choice=True):
    if random_choice:
        f = lambda n: [(lambda n, arr: n if arr == [] else np.random.choice(arr))(n * i / sn,
                    range(int(n * i / sn), max(int(n * i / sn) + 1, int(n * (i + 1) / sn))))
                    for i in range(sn)]
    else:
        f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn,
                    range(int(n * i / sn), max(int(n * i / sn) + 1, int(n * (i + 1) / sn))))
                    for i in range(sn)]
    return f(clip)

def cosine_scheduler(base_value, final_value, epochs):
    iters = np.arange(epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    return schedule

def cosine_scheduler_func(base_value, final_value, iters, epochs):
    schedule = lambda x: final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * x / epochs))
    return schedule(iters)

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
    return loaded_object

def yield_tokens(file_path):
    with io.open(file_path, encoding='utf-8') as f:
        for line in f:
            yield line.strip().split()

@torch.no_grad()
def concat_all_gather(tensor):
    """All-gather for tensors (no grad)."""
    world = get_world_size()
    if world == 1:
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(world)]
    tdist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


# -----------------------------
# DeepSpeed config / init
# -----------------------------

def get_train_ds_config(offload,
                        dtype,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name="",
                        args=''):
    device = "cpu" if offload else "none"
    data_type = "fp16"
    dtype_config = {"enabled": False}

    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {"enabled": True, "loss_scale_window": 100}
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}

    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {"device": device},
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }

    steps_per_print = getattr(args, "print_freq", 50) if hasattr(args, "__dict__") else 50

    return {
        "steps_per_print": steps_per_print,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        },
    }

def init_deepspeed(args, model, optimizer, lr_scheduler):
    ds_config = get_train_ds_config(
        offload=args.offload,
        dtype=args.dtype,
        stage=args.zero_stage,
        args=args,
    )
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    ds_config['gradient_clipping'] = args.gradient_clipping

    print("Using deepspeed to train...")
    print("Initializing deepspeed...")
    _wrapped_model, _optimizer, _, _lr_sched = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True
    )
    return _wrapped_model, _optimizer, _lr_sched


# -----------------------------
# Repro/args
# -----------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True  # dynamic input dims
    cudnn.benchmark = False     # dynamic input dims

def get_args_parser():
    parser = argparse.ArgumentParser('Uni-Sign scripts', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=8, type=int)
    parser.add_argument('--gradient-clipping', default=1., type=float)
    parser.add_argument('--epochs', default=20, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)

    # logging / dataloader
    parser.add_argument('--print-freq', default=50, type=int, help='Steps between logs')
    parser.add_argument('--prefetch-factor', default=4, type=int, help='DataLoader prefetch factor')
    parser.add_argument('--persistent-workers', action='store_true', help='Use persistent DataLoader workers')

    # Finetuning
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Optimizer
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.98], use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay')

    # Scheduler
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR')
    parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR')
    parser.add_argument('--warmup-epochs', type=float, default=0, metavar='N')

    # Base params
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    # DeepSpeed features
    parser.add_argument('--offload', action='store_true', help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp16', 'bf16'], help='Training data type')
    parser.add_argument('--zero_stage', type=int, default=2, help='ZeRO optimization stage')
    parser.add_argument('--compute_fp32_loss', action='store_true',
                        help='If specified, calculate loss in fp32 for low precision dtypes.')

    parser.add_argument('--quick_break', type=int, default=0, help='save ckpt per quick_break step')

    # RGB branch
    parser.add_argument('--rgb_support', action='store_true')

    # Pose length
    parser.add_argument("--max_length", default=256, type=int)

    # dataset / task
    parser.add_argument("--dataset", default="CSL_Daily", choices=['CSL_News', "CSL_Daily", "WLASL"])
    parser.add_argument("--task", default="SLT", choices=['SLT', "ISLR", "CSLR"])

    # label smoothing
    parser.add_argument("--label_smoothing", default=0.2, type=float)

    # online inference
    parser.add_argument("--online_video", default="", type=str)

    # ---- W&B flags ----
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='unisign', help='W&B project')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity/org')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='Run name')
    parser.add_argument('--wandb-group', type=str, default=None, help='Group name')
    parser.add_argument('--wandb-tags', nargs='*', default=None, help='Tags list')
    parser.add_argument('--wandb-mode', type=str, default=None, choices=['online', 'offline', 'disabled'], help='W&B mode')
    parser.add_argument('--wandb-id', type=str, default=None, help='Run ID to resume')
    parser.add_argument('--wandb-watch', action='store_true', help='Watch model gradients/params')

    return parser
