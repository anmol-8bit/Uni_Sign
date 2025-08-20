# pre_training.py
import os
import sys
import time
import json
import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from timm.optim import create_optimizer
from transformers import get_scheduler

import utils as utils
from models import Uni_Sign, get_requires_grad_dict
from datasets import S2T_Dataset_news
from SLRT_metrics import translation_performance
from config import *  # train_label_paths, dev_label_paths


def save_on_master_safe(state, path):
    """
    Wrapper to avoid AttributeError if utils.save_on_master does not exist.
    """
    try:
        if hasattr(utils, "save_on_master"):
            utils.save_on_master(state, path)
        else:
            if utils.is_main_process():
                torch.save(state, path)
    except Exception as e:
        print(f"[warn] save_on_master fallback error for {path}: {e}")


def build_loader_news(args, phase: str, aug_progress: float = 0.0):
    """
    Build CSL_News dataset + loader. For train, pass aug_progress in [0,1].
    """
    if phase == 'train':
        dataset = S2T_Dataset_news(path=train_label_paths[args.dataset], args=args, phase='train')
        # Set augmentation progress (will be used in __getitem__)
        dataset.set_augmentation_progress(aug_progress)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    else:
        dataset = S2T_Dataset_news(path=dev_label_paths[args.dataset], args=args, phase='dev')
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        sampler=sampler,
        pin_memory=args.pin_mem,
        drop_last=(phase == 'train'),
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )
    return dataset, sampler, loader


def main(args):
    utils.init_distributed_mode_ds(args)

    print(args)
    utils.set_seed(args.seed)

    # Build dev (fixed) and a temp train loader (for scheduler sizing)
    print("Creating dev dataset/loader:")
    dev_data, dev_sampler, dev_dataloader = build_loader_news(args, phase='dev', aug_progress=0.0)
    print(dev_data)

    print("Creating temp train dataset/loader (for schedule sizing):")
    tmp_train_data, tmp_train_sampler, tmp_train_dataloader = build_loader_news(args, phase='train', aug_progress=0.0)
    print(tmp_train_data)

    print("Creating model:")
    model = Uni_Sign(args=args).cuda().train()

    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')['model']
        ret = model.load_state_dict(state_dict, strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    model_without_ddp = model

    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)

    if args.quick_break <= 0:
        args.quick_break = len(tmp_train_dataloader)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_epochs * len(tmp_train_dataloader) / args.gradient_accumulation_steps),
        num_training_steps=int(args.epochs * len(tmp_train_dataloader) / args.gradient_accumulation_steps),
    )

    # DeepSpeed init
    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)
    model_without_ddp = model.module  # DeepSpeed engine -> underlying module

    # ---- W&B init (rank-0 only) ----
    world_size = utils.get_world_size()
    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size_per_gpu": args.batch_size,
        "grad_accum": args.gradient_accumulation_steps,
        "world_size": world_size,
        "zero_stage": args.zero_stage,
        "dtype": args.dtype,
        "dataset": args.dataset,
        "max_length": args.max_length,
        "print_freq": args.print_freq,
        "num_workers": args.num_workers,
        "prefetch_factor": getattr(args, "prefetch_factor", None),
        "persistent_workers": args.persistent_workers,
        "rgb_support": args.rgb_support,
    }
    # Your utils.init_wandb should accept (args, config=...) per your current codebase
    wandb_run = utils.init_wandb(args, config=config)
    if getattr(args, "wandb_watch", False):
        utils.wandb_watch(wandb_run, model_without_ddp, log="gradients", log_freq=max(1, args.print_freq))

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    max_bleu4 = 0.0
    global_step = 0  # count dataloader iterations

    if args.eval:
        if utils.is_main_process():
            print("📄 test result")
            _ = evaluate(args, dev_dataloader, model, model_without_ddp, wandb_run, global_step)
        return

    # Schedule: ramp augmentation over first ~30% epochs
    aug_ramp_epochs = max(1, int(0.3 * args.epochs))

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(0, args.epochs):
        # progress in [0,1]
        progress = min(1.0, epoch / float(aug_ramp_epochs))
        print(f"[Epoch {epoch}] augmentation progress = {progress:.3f}")

        # Rebuild train dataset/loader so worker processes pick up updated progress
        train_data, train_sampler, train_dataloader = build_loader_news(args, phase='train', aug_progress=progress)
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats, global_step = train_one_epoch(
            args, model, train_dataloader, optimizer, epoch,
            model_without_ddp=model_without_ddp,
            wandb_run=wandb_run,
            step_offset=global_step
        )

        # Save epoch checkpoint
        if output_dir:
            checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
            for checkpoint_path in checkpoint_paths:
                save_on_master_safe({'model': get_requires_grad_dict(model_without_ddp)}, checkpoint_path)

        # Evaluate on fixed dev
        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, wandb_run, global_step)
        print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['bleu4']:.2f}")

        # log per-epoch to W&B
        utils.wandb_log(wandb_run, {
            "epoch": epoch,
            "train/loss": train_stats.get("loss", float('nan')),
            "val/loss": test_stats.get("loss", float('nan')),
            "val/bleu1": test_stats.get("bleu1", float('nan')),
            "val/bleu2": test_stats.get("bleu2", float('nan')),
            "val/bleu3": test_stats.get("bleu3", float('nan')),
            "val/bleu4": test_stats.get("bleu4", float('nan')),
            "val/rouge": test_stats.get("rouge", float('nan')),
            "max_mem_MB": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0,
        }, step=global_step)

        # Track best BLEU-4
        if max_bleu4 < test_stats["bleu4"]:
            max_bleu4 = test_stats["bleu4"]
            if output_dir and utils.is_main_process():
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    save_on_master_safe({'model': get_requires_grad_dict(model_without_ddp)}, checkpoint_path)

        print(f'Max BLEU-4: {max_bleu4:.2f}%')
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch}

        if output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # finish W&B
    if wandb_run is not None and utils.is_main_process():
        try:
            wandb_run.finish()
        except Exception:
            pass


def train_one_epoch(args, model, data_loader, optimizer, epoch, model_without_ddp, wandb_run=None, step_offset=0):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}/{args.epochs}]'
    print_freq = getattr(args, "print_freq", 50)

    optimizer.zero_grad()

    target_dtype = torch.bfloat16 if model.bfloat16_enabled() else None
    dev_index = args.gpu if hasattr(args, "gpu") and torch.cuda.is_available() else torch.cuda.current_device() if torch.cuda.is_available() else None
    device = torch.device(f"cuda:{dev_index}") if dev_index is not None else torch.device("cpu")

    running_loss = torch.zeros((), device=device)
    running_count = 0
    step = step_offset

    for iter_idx, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if (iter_idx + 1) % args.quick_break == 0 and args.output_dir:
            output_dir = Path(args.output_dir)
            checkpoint_paths = [output_dir / f'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                save_on_master_safe({'model': get_requires_grad_dict(model_without_ddp)}, checkpoint_path)

        # Move to device; only cast float tensors
        for k, v in list(src_input.items()):
            if isinstance(v, torch.Tensor):
                v = v.to(device, non_blocking=True)
                if target_dtype is not None and v.is_floating_point():
                    v = v.to(target_dtype)
                src_input[k] = v

        stack_out = model(src_input, tgt_input)
        total_loss = stack_out['loss']

        if not torch.isfinite(total_loss.detach()).all():
            print("Loss is NaN/Inf, stopping training")
            sys.exit(1)

        model.backward(total_loss)
        model.step()

        running_loss += total_loss.detach()
        running_count += 1
        step += 1

        # periodic log (averaged since last print)
        if (iter_idx + 1) % print_freq == 0 or (iter_idx + 1) == len(data_loader):
            loss_val = (running_loss / max(1, running_count)).float().item()
            current_lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss_val, lr=current_lr)
            utils.wandb_log(wandb_run, {
                "train/loss": loss_val,
                "train/lr": current_lr,
            }, step=step)
            running_loss.zero_()
            running_count = 0

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return ({k: meter.global_avg for k, meter in metric_logger.meters.items()}, step)


def evaluate(args, data_loader, model, model_without_ddp, wandb_run=None, step_for_log: int = 0):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = getattr(args, "print_freq", 50)

    target_dtype = torch.bfloat16 if model.bfloat16_enabled() else None
    dev_index = args.gpu if hasattr(args, "gpu") and torch.cuda.is_available() else torch.cuda.current_device() if torch.cuda.is_available() else None
    device = torch.device(f"cuda:{dev_index}") if dev_index is not None else torch.device("cpu")

    with torch.no_grad():
        preds = []
        refs = []

        for _, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            for k, v in list(src_input.items()):
                if isinstance(v, torch.Tensor):
                    v = v.to(device, non_blocking=True)
                    if target_dtype is not None and v.is_floating_point():
                        v = v.to(target_dtype)
                    src_input[k] = v

            stack_out = model(src_input, tgt_input)
            total_loss = stack_out['loss']
            metric_logger.update(loss=total_loss.detach().float().item())

            output = model_without_ddp.generate(stack_out, max_new_tokens=100, num_beams=4)
            decoded = model_without_ddp.mt5_tokenizer.batch_decode(output, skip_special_tokens=True)

            preds.extend(decoded)
            refs.extend(tgt_input['gt_sentence'])

    if args.dataset == 'CSL_News':
        preds = [' '.join(list(r.replace(" ", '').replace("\n", ''))) for r in preds]
        refs  = [' '.join(list(r.replace("，", ',').replace("？", "?").replace(" ", ''))) for r in refs]

    bleu_dict, rouge_score = translation_performance(refs, preds)
    for k, v in bleu_dict.items():
        metric_logger.meters[k].update(v)
    metric_logger.meters['rouge'].update(rouge_score)

    metric_logger.synchronize_between_processes()
    print('* BLEU-4 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'.format(
        top1=metric_logger.bleu4, losses=metric_logger.loss))

    # log validation summary to W&B
    utils.wandb_log(wandb_run, {
        "val/loss": metric_logger.loss.global_avg,
        "val/bleu1": metric_logger.bleu1.global_avg if 'bleu1' in metric_logger.meters else float('nan'),
        "val/bleu2": metric_logger.bleu2.global_avg if 'bleu2' in metric_logger.meters else float('nan'),
        "val/bleu3": metric_logger.bleu3.global_avg if 'bleu3' in metric_logger.meters else float('nan'),
        "val/bleu4": metric_logger.bleu4.global_avg if 'bleu4' in metric_logger.meters else float('nan'),
        "val/rouge": metric_logger.rouge.global_avg if 'rouge' in metric_logger.meters else float('nan'),
    }, step=step_for_log)

    # Optionally: log a few predictions (rank-0)
    if wandb_run is not None and utils.is_main_process():
        try:
            import wandb
            k = min(8, len(preds))
            table = wandb.Table(data=[[refs[i], preds[i]] for i in range(k)], columns=["ref", "pred"])
            wandb_run.log({"samples": table}, step=step_for_log)
        except Exception:
            pass

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import argparse
    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
