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
from data import create_dataset, create_sampler, create_test_loader
from data.eval_tablip_datasets import tablip_collate_fn
from data.utils import save_result
import utils
from utils import (
    custom_format_html,
    decode_OTSL_seq,
)



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
        dr_coords = batch['dr_coords']
        valid_coord_lens = batch['valid_coord_lens']
        html_with_content = batch['html_with_content']
        cell_texts = batch['cell_texts']
        file_names = batch['file_names']

        image = image.to(device)

        with torch.no_grad():
            preds = model.inference(image, dr_coords, valid_coord_lens)

        B = image.size(0)
        total_similarity_in_batch = 0.0

        pred_sequences = preds["output_sequences"] # OTSL token ids list
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
            answer_string, _ = answer_html, None if '<html>' in answer_html else custom_format_html(answer_html, model.tokenizer) 
            

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
    test_dataset = create_dataset('eval_tablip', config)   
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(test_dataset, [False], num_tasks, global_rank)         
    else:
        samplers = None
    
    test_loader = create_test_loader(
        test_dataset, 
        samplers,
        batch_size=config['batch_size'],
        num_worker=config['num_worker'],
        collate_fn=tablip_collate_fn, 
    )

    dataloaders = { "test": test_loader }

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

    ######## Load checkpoint ########
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        if 'optimizer' in checkpoint: pass
        if 'epoch' in checkpoint: pass           
        if 'scheduler' in checkpoint: pass
        print('resume checkpoint from %s'%args.checkpoint)

    ######## Training ########  
    print("Start testing")
    start_time = time.time()    

    results = evaluation(model, dataloaders["test"], device, config)
    result_file = save_result(results, args.result_dir, 'tablip_result')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/eval.yaml') 
    parser.add_argument('--output_dir', default='output/posttrain_test')
    parser.add_argument('--checkpoint', default='')
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