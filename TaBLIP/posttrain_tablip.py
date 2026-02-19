import argparse
import os
from ruamel.yaml import YAML
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from functools import partial
from Levenshtein import distance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from models.tablip_posttrain import tablip_posttrain, TaBLIP_posttrain
from data import create_dataset, create_sampler, create_loader
from data.posttrain_tablip_datasets import tablip_collate_fn, tablip_collate_fn_test
from data.utils import save_result
import utils
from utils import (
    cosine_lr_schedule, 
    cosine_schedule_with_warmup, 
    configure_optimizer_weight_decay,
    custom_format_html,
    decode_OTSL_seq,
    compute_grad_norm,
)



def train(model, dataloaders, optimizer, lr_scheduler, epoch, device, valid_step, config):

    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ptr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config.get('use_bbox_HiMulConET', False):
        if config.get('use_RowWise_contLearning', False):
            metric_logger.add_meter('loss_rowwise', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        if config.get('use_ColWise_contLearning', False):    
            metric_logger.add_meter('loss_colwise', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('pointer_acc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))


    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    for i, batch in enumerate(metric_logger.log_every(dataloaders["train"], print_freq, header)):
        
        image = batch['image']
        otsl_seq = batch['OTSL_seq']
        dr_coords = batch['dr_coords']
        valid_coord_lens = batch['valid_coord_lens']
        pointer_labels = batch['pointer_labels']
        pointer_mask_labels = batch['pointer_mask_labels']
        chosen_bbox_coeff_tensors = batch['chosen_bbox_coeff_tensors']

        image = image.to(device, non_blocking=True)

        output = model(
            image, 
            otsl_seq, 
            dr_coords, 
            valid_coord_lens,
            pointer_labels,
            pointer_mask_labels,
            chosen_bbox_coeff_tensors,
        )      

        loss = output["loss_lm"]
        output["loss"] = output["loss"].detach()
        output["loss_ptr"] = output["loss_ptr"].detach()
        output["pointer_acc"] = output["pointer_acc"].detach()
        if config.get('use_bbox_HiMulConET', False):
            output["loss_rowwise"] = output["loss_rowwise"].detach()
            output["loss_colwise"] = output["loss_colwise"].detach()

        optimizer.zero_grad()
        loss.backward()
        if config['grad_clip']:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])
        grad_norm = compute_grad_norm(model)
        optimizer.step()
        lr_scheduler.step()
        
        metric_logger.update(loss=output["loss"].item())
        metric_logger.update(loss_lm=output["loss_lm"].item())
        metric_logger.update(loss_ptr=output["loss_ptr"].item())
        metric_logger.update(pointer_acc=output["pointer_acc"].item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_norm)
        if config.get('use_bbox_HiMulConET', False):
            metric_logger.update(loss_rowwise=output["loss_rowwise"].item())
            metric_logger.update(loss_colwise=output["loss_colwise"].item())

        if (i + 1) % valid_step == 0:
            meters, config['best_pointer_acc'] = validation(
                model, dataloaders["test"], epoch, device, config)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg()) 
    meters = {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 
    return meters, config['best_pointer_acc']


@torch.no_grad()
def validation(model, data_loader, epoch, device, config):
    """Validation step"""
    model.eval()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ptr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_rowwise', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_colwise', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('pointer_acc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    
    header = 'Valid Model: [{}]'.format(epoch)
    print_freq = 100
    best_pointer_acc = config['best_pointer_acc']

    with torch.no_grad():
        for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

            image = batch['image']
            otsl_seq = batch['OTSL_seq']
            dr_coords = batch['dr_coords']
            valid_coord_lens = batch['valid_coord_lens']
            pointer_labels = batch['pointer_labels']
            pointer_mask_labels = batch['pointer_mask_labels']
            chosen_bbox_coeff_tensors = batch['chosen_bbox_coeff_tensors']
            html_with_content = batch['html_with_content']
            cell_texts = batch['cell_texts']
            file_names = batch['file_names']

            image = image.to(device, non_blocking=True)

            output = model(
                image, 
                otsl_seq, 
                dr_coords,
                valid_coord_lens,
                pointer_labels,
                pointer_mask_labels,
                chosen_bbox_coeff_tensors,
            )

            metric_logger.update(loss=output["loss"].item())
            metric_logger.update(loss_lm=output["loss_lm"].item())
            metric_logger.update(loss_ptr=output["loss_ptr"].item())
            metric_logger.update(pointer_acc=output["pointer_acc"].item())
            metric_logger.update(loss_rowwise=output["loss_rowwise"].item())
            metric_logger.update(loss_colwise=output["loss_colwise"].item())


    model.train()
    metric_logger.synchronize_between_processes()
    print("Valid Averaged stats:", metric_logger.global_avg())   
    meters = {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 

    now = float(meters["loss_lm"])
    if now < best_pointer_acc:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_pointer_acc.pth'))
        best_pointer_acc = now
        print("Valid Best Loss LM: ", best_pointer_acc)
    return meters, best_pointer_acc
    


@torch.no_grad()
def evaluation(model: TaBLIP_posttrain, data_loader, device, config):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('string_similarity', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    result_collection = {}
    header = 'Evaluation Model: '
    print_freq = 50
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image = batch['image']
        otsl_seq = batch['OTSL_seq']
        dr_coords = batch['dr_coords']
        valid_coord_lens = batch['valid_coord_lens']
        pointer_labels = batch['pointer_labels']
        pointer_mask_labels = batch['pointer_mask_labels']
        chosen_bbox_coeff_tensors = batch['chosen_bbox_coeff_tensors']
        html_with_content = batch['html_with_content']
        cell_texts = batch['cell_texts']
        file_names = batch['file_names']

        image = image.to(device)

        with torch.no_grad():
            preds = model.inference(image, dr_coords, valid_coord_lens)

        B = image.size(0)
        total_similarity_in_batch = 0.0

        pred_sequences = preds["output_sequences"]
        gt_htmls = html_with_content
        all_cell_texts = cell_texts

        for i in range(B):
            token_id_seq = pred_sequences[i]
            token_seq = model.tokenizer.convert_ids_to_tokens(token_id_seq)
            if token_seq[0] == '[DEC]': token_seq = token_seq[1:]
            if token_seq[0] == '[SEP]': token_seq = token_seq[:-1]

            cell_text_data = all_cell_texts[i].split("<special_cell_text_sep>")
            
            pred_html = decode_OTSL_seq(
                otsl_token_seq=token_seq,
                pointer_tensor=preds["text_to_dr_coord"][i],
                cell_text_data=cell_text_data,
            )

            answer_html = gt_htmls[i]

            pred_string, _ = custom_format_html(pred_html, model.tokenizer)
            answer_string, _ = custom_format_html(answer_html, model.tokenizer)

            edit_distance = distance(pred_string, answer_string) / max(
                len(pred_string), len(answer_string)
            )
            similarity = 1 - edit_distance
            total_similarity_in_batch += similarity

            curr_filename = file_names[i]
            assert curr_filename not in result_collection, f"Duplicate filename: {curr_filename}"
            result_collection[curr_filename] = {
                "pred_string": pred_string,
                "answer_string": answer_string,
                "similarity": similarity,
                "token_pred": token_seq,
                "token_gold": otsl_seq[i],
            }

        avg_similarity = total_similarity_in_batch / B if B > 0 else 0.0
        metric_logger.update(string_similarity=avg_similarity)
        torch.cuda.empty_cache()

    metric_logger.synchronize_between_processes()
    print("Evaluation Averaged stats:", metric_logger.global_avg())

    torch.cuda.empty_cache()
    model.train()
    return result_collection, float(metric_logger.global_avg().split(':')[1].strip())



def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    ######## Dataset ######## 
    print("Creating vqa datasets")
    datasets = create_dataset('posttrain_tablip', config)   
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader(
        datasets, samplers,
        batch_size=[config['batch_size_train'], config['batch_size_test']],
        num_workers=[4,4], is_trains=[True, False], 
        collate_fns=[tablip_collate_fn, tablip_collate_fn_test]) 
    dataloaders = {
        "train":train_loader,
        "test": test_loader
    }

    ######## Model ########
    print("Creating model")
    model = tablip_posttrain(pretrained=config['pretrained'], config=config)
    model = model.to(device)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Params: {:.2f}MB'.format(total_params * 4 / (1024**2)) )
    
    ######## Optimizer and Lr_scheduler ########
    
    optim_params = configure_optimizer_weight_decay(model, config['weight_decay'])
    optimizer = torch.optim.AdamW(params=optim_params, lr=config['init_lr'])
    
    total_step = config['max_epoch'] * len(train_loader) + 200
    warmup_step = int(total_step * config.get('warmup_ratio', 0.01) )
    min_ratio = config['min_lr'] / config['init_lr']
    print(f"Total training steps: {total_step}")
    print(f"Warmup steps: {warmup_step}")

    lr_schedule_func = partial(
        cosine_schedule_with_warmup, warmup=warmup_step, min_ratio=min_ratio, total_step=total_step)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer, lr_lambda=lr_schedule_func)

    best_epoch = 0
    best_score = 0                  
    start_epoch = 0                 
    valid_step = args.valid_step    

    ######## Load checkpoint ########
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:   
            start_epoch = checkpoint['epoch'] + 1                
        if 'scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
        print('resume checkpoint from %s'%args.checkpoint)

    ######## Training ########  
    print("Start training")
    start_time = time.time()    
    for epoch in range(start_epoch, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                dataloaders["train"].sampler.set_epoch(epoch)

            train_stats, config["best_pointer_acc"] = train(
                model, dataloaders, optimizer, lr_scheduler, epoch, device, valid_step, config) 
        else:
            break        
        
        if utils.is_main_process():     
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
                    
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))
        
        if (epoch + 1) % config['eval_epoch'] == 0 or (epoch + 1) == config['max_epoch']:
            results, score = evaluation(model, dataloaders["test"], device, config)
            if score > best_score: 
                result_file = save_result(results, args.result_dir, 'tablip_result')
                best_score = score

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            dist.barrier()

    if args.evaluate:
        results = evaluation(model, dataloaders["test"], device, config)
        result_file = save_result(results, args.result_dir, 'tablip_result')
                      

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/posttrain.yaml') 
    parser.add_argument('--output_dir', default='output/posttrain_test')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--valid_step', default=50000, type=int)
    parser.add_argument('--evaluate', action="store_true")
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    yaml = YAML(typ='safe') 
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    writer = SummaryWriter('runs/experiment_name')
    
    main(args, config)