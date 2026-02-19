import math
import torch.nn as nn

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):        
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    


def cosine_schedule_with_warmup(
    step: int, *, warmup: int, min_ratio: float, total_step: int, cycle: float = 0.5):
    """返回 µ , 与 lr 相乘"""

    if step < warmup:
        if step == 0: step = 1
        return float(step) / float(max(1, warmup))

    if step >= total_step: step = total_step

    progress = float(step - warmup) / float(max(1, total_step - warmup))
    progress = min(progress, 1.0)
    return max(
        min_ratio, 0.5 * (1.0 + math.cos(math.pi * float(cycle) * 2.0 * progress))
    )


def configure_optimizer_weight_decay(
    model: nn.Module, weight_decay: float
):
    weight_decay_blacklist = (nn.LayerNorm, nn.BatchNorm2d, nn.Embedding)

    skip_list = set()
    if hasattr(model, "no_weight_decay"):
        skip_list = model.no_weight_decay()
    decay = set()
    no_decay = set()

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if n.endswith("bias"):
            no_decay.add(n)
            continue
        if n in skip_list:
            no_decay.add(n)
            continue

        module_name = '.'.join(n.split('.')[:-1])
        try:
            module = model.get_submodule(module_name)
            if isinstance(module, weight_decay_blacklist):
                no_decay.add(n)
        except AttributeError:
            print("Fall")
            pass

    param_dict = {pn: p for pn, p in model.named_parameters()}
    decay = param_dict.keys() - no_decay

    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    return optim_groups


def custom_format_html(html_string, tokenizer):
    """Custom format html string"""
    tokens_to_remove = [
        tokenizer.bos_token,
        tokenizer.eos_token,
        tokenizer.pad_token,
        "<s_answer>",
        "</s_answer>",
    ]
    for token in tokens_to_remove:
        html_string = html_string.replace(token, "")

    html_seq = "<html><body><table>" + html_string + "</table></body></html>"

    return html_string, html_seq


def decode_OTSL_seq(otsl_token_seq, pointer_tensor, cell_text_data):
    """Decode otsl token seq from token seq and pointer prediction

    Args:
        otsl_token_seq List[str]: token sequence, don't have [DEC] and [SEP]
        point_prediction torch.Tensor: pointer prediction
        cell_text_data List[str]: cell text data

    Returns:
        output_seq_tokens str: html sequence
    """

    cell_text = None
    OTSL_full_compilation = []
    OTSL_row_compilation = []
    curr_column_index = 0

    for data_ind, token in enumerate(otsl_token_seq):
        if token == "C-tag":
            mapping_mask = pointer_tensor[data_ind]  # (bbox_token_cnt,)


            coord_indices = torch.nonzero(mapping_mask).squeeze(-1)  # (num_of_coords,)
            if len(coord_indices) == 0:
                cell_text = None
            else:
                indices_list = coord_indices.tolist()
                for coord_ind in indices_list:
                    if coord_ind == 0: continue
                    elif coord_ind > len(cell_text_data): continue
                    else:
                        if cell_text is None:
                            cell_text = cell_text_data[coord_ind - 1]
                        else:
                            cell_text += " " + cell_text_data[coord_ind - 1]

            OTSL_row_compilation.append([1, 0, 0, cell_text])
            curr_column_index += 1
            cell_text = None
        elif token == "NL-tag":
            OTSL_full_compilation.append(OTSL_row_compilation)
            OTSL_row_compilation = []
            curr_column_index = 0
        elif token == "L-tag":
            for col_i in range(len(OTSL_row_compilation)):
                col_i_value = OTSL_row_compilation[-1 - col_i]
                if col_i_value is not None:
                    col_i_value[2] += 1
                    break
            OTSL_row_compilation.append(None)
            curr_column_index += 1

        elif token == "U-tag":
            for row_i in range(len(OTSL_full_compilation)):
                row_i_value = OTSL_full_compilation[-1 - row_i]
                if (
                    curr_column_index < len(row_i_value)
                    and row_i_value[curr_column_index] is not None
                ):
                    row_i_value[curr_column_index][1] += 1
                    break

            OTSL_row_compilation.append(None)
            curr_column_index += 1
        elif token == "X-tag":
            OTSL_row_compilation.append(None)
            curr_column_index += 1
            continue
        else: continue

    if len(OTSL_row_compilation) > 0:
        OTSL_full_compilation.append(OTSL_row_compilation)

    OTSL_full_compilation = [
        item for sublist in OTSL_full_compilation for item in sublist
    ]
    output_html_seq = "<tr>"
    current_data_index = 0
    for i, token in enumerate(otsl_token_seq):
        if token in ["L-tag", "U-tag", "X-tag"]:
            current_data_index += 1
            continue
        elif token == "C-tag":
            cell_info = OTSL_full_compilation[current_data_index]
            if cell_info is not None:
                if cell_info[1] == 0 and cell_info[2] == 0:
                    if cell_info[3] is None:
                        output_html_seq += "<td></td>"
                    else:
                        output_html_seq += "<td>" + cell_info[3] + "</td>"

                elif cell_info[1] == 0:
                    if cell_info[3] is None:
                        output_html_seq += '<td colspan="%s"></td>' % (cell_info[2] + 1)
                    else:
                        output_html_seq += '<td colspan="%s">' % (cell_info[2] + 1) + cell_info[3] + "</td>"

                elif cell_info[2] == 0:
                    if cell_info[3] is None:
                        output_html_seq += '<td rowspan="%s"></td>' % (cell_info[1] + 1)
                    else:
                        output_html_seq += '<td rowspan="%s">' % (cell_info[1] + 1) + cell_info[3] + "</td>"

                else:
                    if cell_info[3] is None:
                        output_html_seq += '<td rowspan="%s" colspan="%s"></td>' % (cell_info[1] + 1, cell_info[2] + 1)
                    else:
                        output_html_seq += (
                            '<td rowspan="%s" colspan="%s">' % (cell_info[1] + 1, cell_info[2] + 1)
                            + cell_info[3]
                            + "</td>"
                        )

            current_data_index += 1

        elif token == "NL-tag":
            output_html_seq += "</tr><tr>"
        
        elif token in ["[DEC]", "[SEP]"]: continue

        else:
            if token == "▁":
                token_to_add = " "
                output_html_seq += token_to_add
            else:
                token_to_add = token.replace("▁", "")
                output_html_seq += token_to_add

    tmp_split = output_html_seq.rsplit("<tr>", 1)
    output_html_seq = tmp_split[0] + tmp_split[1]

    output_html_seq = output_html_seq.replace("<pad>", "")

    return output_html_seq


def compute_grad_norm(model) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm




import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
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
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


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
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
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
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
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
            header, total_time_str, total_time / len(iterable)))
        

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
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
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)        
        
        