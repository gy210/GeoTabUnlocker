import os
import time
import datetime
import json
from pathlib import Path
from ruamel.yaml import YAML
from typing import List, Tuple

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from geounlock.tablip_model.tablip import tablip, TaBLIP
from geounlock.utils import (
    custom_format_html,
    decode_OTSL_seq,
)


class TaBLIP_pipeline:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.image_size = config.get('image_size', 448)
        self.bbox_token_cnt = config.get('bbox_token_cnt', 448)
        self.num_beams = config.get('num_beams', 1)

        print("正在初始化 TaBLIP 模型 ...")
        self.model = tablip(pretrained=config['pretrained'], config=config)
        self.model = self.model.to(device)

        self.transform = transformer(self.image_size)


    def predict(self, image: List[Image.Image], ocr_data: list):
        assert len(image) == len(ocr_data)

        image, org_img_sizes = self.get_image_tensor(image)

        new_img_sizes = [(self.image_size, self.image_size)] * len(org_img_sizes)
        dr_coords, valid_coord_lens, cell_texts = self.get_dr_coords(
            ocr_data, org_img_sizes, new_img_sizes
        )
        
        with torch.no_grad():
            image = image.to(self.device)
            dr_coords = dr_coords.to(self.device)
            valid_coord_lens = valid_coord_lens.to(self.device)
            preds = self.model.inference(
                image=image,
                dr_coords=dr_coords,
                valid_coord_lens=valid_coord_lens,
                num_beams=self.num_beams
            )

        structured_results = self._postprocess(preds, cell_texts)
        
        return {
            'html': structured_results,
        }


    def get_image_tensor(self, image):
        org_img_size = [img.size for img in image]
        image = [self.transform(img) for img in image]
        image = torch.stack(image, dim=0)

        return image, org_img_size


    def get_dr_coords(self, ocr_data, org_img_sizes, new_img_sizes):
        dr_coords = []
        valid_coord_lens = []
        cell_texts = []

        padding_coord = [
            new_img_sizes[0][0] + 3,
            new_img_sizes[0][1] + 3,
            new_img_sizes[0][0] + 3,
            new_img_sizes[0][1] + 3,
        ]

        for item, org_size, new_size in zip(ocr_data, org_img_sizes, new_img_sizes):
            
            texts = item['rec_texts']
            coords = torch.from_numpy(item['rec_boxes'])

            coords, texts = self.sort_coords_and_texts(coords, texts)

            coords_pad = rescale_pad_coords(
                coords=coords,
                padding_coord=padding_coord,
                bbox_token_cnt=self.bbox_token_cnt,
                org_img_size=org_size,
                new_img_size=new_size,
            ).round().to(torch.int32)

            dr_coords.append(coords_pad)
            valid_coord_lens.append(min(len(coords), self.bbox_token_cnt - 1))
            cell_texts.append("<special_cell_text_sep>".join(texts))

        dr_coords = torch.stack(dr_coords, dim=0)
        valid_coord_lens = torch.tensor(valid_coord_lens, dtype=torch.int16)

        return dr_coords, valid_coord_lens, cell_texts
    

    def sort_coords_and_texts(
        self, coords: torch.Tensor, texts: List[str]
    ) -> Tuple[torch.Tensor, List[str]]:
        num_box = coords.shape[0]
        if num_box <= 1: return coords, texts
        assert num_box == len(texts), "coord和text的数量必须匹配！"
        
        y_min_sort_index = torch.argsort(coords[:, 1])
        coords = coords[y_min_sort_index]
        texts = [texts[i] for i in y_min_sort_index]

        heights = coords[:, 3] - coords[:, 1]
        valid_heights = heights[heights > 0]
        if valid_heights.numel() > 0:
            line_height_threshold = torch.median(valid_heights).item()
        else:
            line_height_threshold = 10.0

        line_indices = torch.zeros(num_box, dtype=torch.long, device=coords.device)
        last_y_center = (coords[0, 1] + coords[0, 3]) / 2.0
        cur_line = 0
        
        for i in range(1, num_box):
            current_y_center = (coords[i, 1] + coords[i, 3]) / 2.0
            
            if abs(current_y_center - last_y_center) > line_height_threshold * 0.5:
                cur_line += 1
                last_y_center = current_y_center

            line_indices[i] = cur_line

        sort_key = line_indices * 10000 + coords[:, 0]
        sort_indices = torch.argsort(sort_key)
        
        sorted_coords = coords[sort_indices]
        sorted_texts = [texts[i] for i in sort_indices]
        
        return sorted_coords, sorted_texts

    
    def _postprocess(self, preds, cell_texts):
        result_collection = []

        pred_sequences = preds["output_sequences"] # OTSL token ids list
        all_cell_texts = cell_texts
        B = pred_sequences.size(0)

        for i in range(B):
            token_id_seq = pred_sequences[i]
            token_seq = self.model.tokenizer.convert_ids_to_tokens(token_id_seq)
            if token_seq[0] == '[DEC]': token_seq = token_seq[1:]
            if token_seq[0] == '[SEP]': token_seq = token_seq[:-1]

            cell_text_data = all_cell_texts[i].split("<special_cell_text_sep>")
            
            pred_html = decode_OTSL_seq(
                otsl_token_seq=token_seq,
                pointer_tensor=preds["text_to_dr_coord"][i],
                cell_text_data=cell_text_data,
            )

            pred_string, _ = custom_format_html(pred_html, self.model.tokenizer)

            result_collection.append(pred_string)
        
        return result_collection



def transformer(image_size: int = 448):
    normalize = transforms.Normalize((0.86597056, 0.88463002, 0.87491087), (0.20686628, 0.18201602, 0.18485524))    
    transform_list = [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ]
    return transforms.Compose([t for t in transform_list if t is not None])


def rescale_pad_coords(
    coords: List[List], padding_coord: int = -1, bbox_token_cnt: int = None,
    org_img_size: Tuple = None, new_img_size: List = None,
) -> torch.tensor:
    def rescale_coords(coords, org_img_size, new_img_size):
        W_o, H_o = org_img_size
        W_t, H_t = new_img_size
        
        X = torch.clamp(coords[:, [0, 2]], min=0, max=W_o - 1)
        Y = torch.clamp(coords[:, [1, 3]], min=0, max=H_o - 1)

        xmin = torch.min(X[:, 0], X[:, 1])
        xmax = torch.max(X[:, 0], X[:, 1])
        ymin = torch.min(Y[:, 0], Y[:, 1])
        ymax = torch.max(Y[:, 0], Y[:, 1])

        coords = torch.stack([xmin, ymin, xmax, ymax], dim=1)
        assert W_o and H_o
        
        x_scale = W_t / W_o
        y_scale = H_t / H_o
        scale_factors = torch.tensor(
            [x_scale, y_scale, x_scale, y_scale], 
            device=coords.device, 
            dtype=torch.float32
        )
        return (coords * scale_factors)
    

    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, dtype=torch.float32)
    else:
        coords = coords.to(torch.float32)
    
    assert coords.shape[0] > 0 and coords.shape[1] == 4
    coords = rescale_coords(coords, org_img_size, new_img_size)

    if bbox_token_cnt is None:
        return coords
    
    coords_len = coords.shape[0]

    if coords_len > (bbox_token_cnt - 1):
        coords = coords[:bbox_token_cnt - 1]
    
    elif coords_len < (bbox_token_cnt - 1):
        if isinstance(padding_coord, list):
            padding_coord = torch.as_tensor(
                padding_coord, dtype=torch.float32).unsqueeze(0)
        else:
            padding_coord = torch.full(
                (1, 4), fill_value=padding_coord, dtype=torch.float32)
            
        pad_size = bbox_token_cnt - coords_len - 1
        coords = torch.cat(
            (coords, padding_coord.expand(pad_size, -1)),
            dim=0
        )

    assert coords.shape[0] == (bbox_token_cnt - 1)
    return coords