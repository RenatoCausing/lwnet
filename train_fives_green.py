"""Train WNet on preprocessed FIVES dataset with green channel only.
Saves model checkpoint and metrics after EVERY cycle.
"""

import sys
import json
import os
import argparse
from shutil import copyfile
import os.path as osp
from datetime import datetime
import operator
from tqdm import tqdm
import numpy as np
import torch
from models.get_model import get_arch

from utils.get_loaders import get_train_val_loaders
from utils.evaluation import evaluate, ewma
from utils.model_saving_loading import save_model, str2bool, load_model
from utils.reproducibility import set_seeds

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--csv_train', type=str, default='data/FIVES_processed/train.csv', help='path to training data csv')
parser.add_argument('--csv_val', type=str, default='data/FIVES_processed/test.csv', help='path to validation/test data csv')
parser.add_argument('--model_name', type=str, default='wnet', help='architecture (wnet, big_wnet, unet, big_unet)')
parser.add_argument('--batch_size', type=int, default=8, help='batch Size')
parser.add_argument('--grad_acc_steps', type=int, default=0, help='gradient accumulation steps (0)')
parser.add_argument('--min_lr', type=float, default=1e-8, help='minimum learning rate')
parser.add_argument('--max_lr', type=float, default=0.01, help='maximum learning rate')
parser.add_argument('--cycle_lens', type=str, default='20/50', help='cycling config (nr cycles/cycle len)')
parser.add_argument('--metric', type=str, default='auc', help='which metric to use for monitoring progress (tr_auc/auc/loss/dice)')
parser.add_argument('--im_size', help='image size, e.g., 512 or 512,512', type=str, default='512')
parser.add_argument('--in_c', type=int, default=1, help='channels in input images (1 for green channel)')
parser.add_argument('--save_path', type=str, default='fives_wnet', help='path to save model')
parser.add_argument('--save_every_cycle', type=str2bool, nargs='?', const=True, default=True, help='save checkpoint after every cycle')
parser.add_argument('--num_workers', type=int, default=0, help='number of parallel workers for data loading')
parser.add_argument('--device', type=str, default='cuda:0', help='device to run on (cpu or cuda:0)')
parser.add_argument('--seed', type=int, default=42, help='random seed')


def compare_op(metric):
    '''Returns comparison operator and initial worst value for the metric'''
    if metric == 'auc':
        return operator.gt, 0
    elif metric == 'tr_auc':
        return operator.gt, 0
    elif metric == 'dice':
        return operator.gt, 0
    elif metric == 'recall':
        return operator.gt, 0
    elif metric == 'loss':
        return operator.lt, np.inf
    else:
        raise NotImplementedError


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run_one_epoch(loader, model, criterion, optimizer=None, scheduler=None,
                  grad_acc_steps=0, assess=False):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None

    model.train() if train else model.eval()

    if assess:
        logits_all, labels_all = [], []
    n_elems, running_loss, tr_lr = 0, 0, 0

    for i_batch, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        
        if isinstance(logits, tuple):  # wnet returns tuple
            logits_aux, logits = logits
            if model.n_classes == 1:
                loss_aux = criterion(logits_aux, labels.unsqueeze(dim=1).float())
                loss = loss_aux + criterion(logits, labels.unsqueeze(dim=1).float())
            else:
                loss_aux = criterion(logits_aux, labels)
                loss = loss_aux + criterion(logits, labels)
        else:
            if model.n_classes == 1:
                loss = criterion(logits, labels.unsqueeze(dim=1).float())
            else:
                loss = criterion(logits, labels)

        if train:
            (loss / (grad_acc_steps + 1)).backward()
            tr_lr = get_lr(optimizer)
            if i_batch % (grad_acc_steps + 1) == 0:
                optimizer.step()
                for _ in range(grad_acc_steps + 1):
                    scheduler.step()
                optimizer.zero_grad()
        
        if assess:
            logits_all.extend(logits)
            labels_all.extend(labels)

        running_loss += loss.item() * inputs.size(0)
        n_elems += inputs.size(0)
        run_loss = running_loss / n_elems

    if assess:
        return logits_all, labels_all, run_loss, tr_lr
    return None, None, run_loss, tr_lr


def train_one_cycle(train_loader, model, criterion, optimizer=None, scheduler=None, 
                    grad_acc_steps=0, cycle=0):
    model.train()
    optimizer.zero_grad()
    cycle_len = scheduler.cycle_lens[cycle]

    with tqdm(range(cycle_len)) as t:
        for epoch in t:
            assess = (epoch == cycle_len - 1)  # only get logits/labels on last epoch of cycle
            tr_logits, tr_labels, tr_loss, tr_lr = run_one_epoch(
                train_loader, model, criterion, optimizer=optimizer,
                scheduler=scheduler, grad_acc_steps=grad_acc_steps, assess=assess
            )
            t.set_postfix(tr_loss_lr="{:.4f}/{:.6f}".format(float(tr_loss), tr_lr))

    return tr_logits, tr_labels, tr_loss


def save_cycle_results(exp_path, cycle, metrics_dict, model, optimizer):
    """Save model checkpoint and metrics for this cycle."""
    cycle_dir = osp.join(exp_path, f'cycle_{cycle:02d}')
    os.makedirs(cycle_dir, exist_ok=True)
    
    # Save model checkpoint for this cycle
    save_model(cycle_dir, model, optimizer)
    
    # Save metrics for this cycle
    metrics_file = osp.join(cycle_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Also save a human-readable version
    metrics_txt = osp.join(cycle_dir, 'metrics.txt')
    with open(metrics_txt, 'w') as f:
        f.write(f"Cycle {cycle} Results\n")
        f.write("=" * 40 + "\n")
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"  Saved cycle {cycle} checkpoint and metrics to {cycle_dir}")


def train_model(model, optimizer, criterion, train_loader, val_loader, scheduler, 
                grad_acc_steps, metric, exp_path, save_every_cycle=True):
    
    n_cycles = len(scheduler.cycle_lens)
    best_auc, best_dice, best_cycle = 0, 0, 0
    is_better, best_monitoring_metric = compare_op(metric)
    
    # Store all cycle results
    all_cycle_results = []

    for cycle in range(n_cycles):
        print(f'\nCycle {cycle + 1}/{n_cycles}')
        print('-' * 50)
        
        # Train one cycle
        tr_logits, tr_labels, tr_loss = train_one_cycle(
            train_loader, model, criterion, optimizer, scheduler, grad_acc_steps, cycle
        )

        # Evaluate at end of cycle
        print('Evaluating...')
        tr_auc, tr_dice, tr_recall = evaluate(tr_logits, tr_labels, model.n_classes)
        del tr_logits, tr_labels
        
        with torch.no_grad():
            vl_logits, vl_labels, vl_loss, _ = run_one_epoch(
                val_loader, model, criterion, assess=True
            )
            vl_auc, vl_dice, vl_recall = evaluate(vl_logits, vl_labels, model.n_classes)
            del vl_logits, vl_labels
        
        print(f'Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f}')
        print(f'Train AUC:  {tr_auc:.4f} | Val AUC:  {vl_auc:.4f}')
        print(f'Train DICE: {tr_dice:.4f} | Val DICE: {vl_dice:.4f}')
        print(f'Train Recall: {tr_recall:.4f} | Val Recall: {vl_recall:.4f}')
        print(f'LR: {get_lr(optimizer):.6f}')

        # Collect metrics for this cycle
        cycle_metrics = {
            'cycle': cycle + 1,
            'train_loss': float(tr_loss),
            'val_loss': float(vl_loss),
            'train_auc': float(tr_auc),
            'val_auc': float(vl_auc),
            'train_dice': float(tr_dice),
            'val_dice': float(vl_dice),
            'train_recall': float(tr_recall),
            'val_recall': float(vl_recall),
            'learning_rate': float(get_lr(optimizer))
        }
        all_cycle_results.append(cycle_metrics)

        # Save checkpoint and metrics for EVERY cycle if enabled
        if save_every_cycle and exp_path is not None:
            save_cycle_results(exp_path, cycle + 1, cycle_metrics, model, optimizer)

        # Check if this is the best model
        if metric == 'auc':
            monitoring_metric = vl_auc
        elif metric == 'tr_auc':
            monitoring_metric = tr_auc
        elif metric == 'loss':
            monitoring_metric = vl_loss
        elif metric == 'dice':
            monitoring_metric = vl_dice
        elif metric == 'recall':
            monitoring_metric = vl_recall
        
        if is_better(monitoring_metric, best_monitoring_metric):
            print(f'*** New best {metric}: {best_monitoring_metric:.4f} -> {monitoring_metric:.4f} ***')
            best_auc, best_dice, best_cycle = vl_auc, vl_dice, cycle + 1
            best_monitoring_metric = monitoring_metric
            
            if exp_path is not None:
                # Save as best model
                best_dir = osp.join(exp_path, 'best_model')
                os.makedirs(best_dir, exist_ok=True)
                save_model(best_dir, model, optimizer)
                with open(osp.join(best_dir, 'best_metrics.json'), 'w') as f:
                    json.dump(cycle_metrics, f, indent=2)
                print(f'  Saved best model to {best_dir}')

    # Save summary of all cycles
    if exp_path is not None:
        summary_file = osp.join(exp_path, 'all_cycles_summary.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'all_cycles': all_cycle_results,
                'best_cycle': best_cycle,
                'best_val_auc': float(best_auc),
                'best_val_dice': float(best_dice)
            }, f, indent=2)
        print(f'\nAll cycle results saved to {summary_file}')

    del model
    torch.cuda.empty_cache()
    return best_auc, best_dice, best_cycle


if __name__ == '__main__':
    args = parser.parse_args()

    # Setup device
    if args.device.startswith("cuda"):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(":", 1)[1]
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")
        print(f'* Training on {args.device}')
        device = torch.device("cuda")
    else:
        device = torch.device(args.device)

    # Reproducibility
    set_seeds(args.seed, args.device.startswith("cuda"))

    # Parse arguments
    model_name = args.model_name
    max_lr, min_lr = args.max_lr, args.min_lr
    bs = args.batch_size
    grad_acc_steps = args.grad_acc_steps
    
    cycle_lens = args.cycle_lens.split('/')
    cycle_lens = list(map(int, cycle_lens))
    if len(cycle_lens) == 2:
        cycle_lens = cycle_lens[0] * [cycle_lens[1]]

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if len(im_size) == 1:
        tg_size = (im_size[0], im_size[0])
    else:
        tg_size = (im_size[0], im_size[1])

    # Setup experiment directory
    save_path = args.save_path
    if save_path == 'date_time':
        save_path = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_path = osp.join('experiments', save_path)
    os.makedirs(experiment_path, exist_ok=True)

    # Save config
    config_file_path = osp.join(experiment_path, 'config.cfg')
    with open(config_file_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Data paths
    csv_train = args.csv_train
    csv_val = args.csv_val

    # Binary segmentation
    n_classes = 1
    label_values = [0, 255]

    print(f"* Creating Dataloaders, batch size = {bs}, workers = {args.num_workers}")
    train_loader, val_loader = get_train_val_loaders(
        csv_path_train=csv_train, 
        csv_path_val=csv_val, 
        batch_size=bs, 
        tg_size=tg_size, 
        label_values=label_values, 
        num_workers=args.num_workers
    )

    print(f'* Instantiating {model_name} model with in_c={args.in_c}')
    model = get_arch(model_name, in_c=args.in_c, n_classes=n_classes)
    model = model.to(device)

    print(f"  Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cycle_lens[0] * len(train_loader), eta_min=min_lr
    )
    setattr(optimizer, 'max_lr', max_lr)
    setattr(scheduler, 'cycle_lens', cycle_lens)

    criterion = torch.nn.BCEWithLogitsLoss()
    print(f'* Loss function: {criterion}')
    
    print(f'* Starting training for {len(cycle_lens)} cycles')
    print(f'* Save every cycle: {args.save_every_cycle}')
    print('-' * 60)

    best_auc, best_dice, best_cycle = train_model(
        model, optimizer, criterion, train_loader, val_loader, scheduler,
        grad_acc_steps, args.metric, experiment_path, 
        save_every_cycle=args.save_every_cycle
    )

    print('\n' + '=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f"Best Val AUC:  {best_auc * 100:.2f}%")
    print(f"Best Val DICE: {best_dice * 100:.2f}%")
    print(f"Best Cycle:    {best_cycle}")
    print(f"Results saved to: {experiment_path}")

    # Save final summary
    with open(osp.join(experiment_path, 'final_results.txt'), 'w') as f:
        f.write(f'Best AUC = {best_auc * 100:.2f}%\n')
        f.write(f'Best DICE = {best_dice * 100:.2f}%\n')
        f.write(f'Best cycle = {best_cycle}\n')
