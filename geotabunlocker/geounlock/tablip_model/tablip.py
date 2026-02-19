import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from transformers import BertTokenizer
import numpy as np

import os
from typing import List
from urllib.parse import urlparse
from timm.models.hub import download_cached_file

from geounlock.tablip_model.unitable_encoder_large import Unitable_Encoder, interpolate_pos_embed
from geounlock.tablip_model.med import BertConfig, BertModel, BertLMHeadModel



class TaBLIP(nn.Module):
    def __init__(self, config, med_config='configs/med_config.json'):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.eval()
        med_config = './configs/med_config.json'

        self.config = config
        self.max_length = config.get("max_length", 256)
        self.bbox_token_cnt = config.get("bbox_token_cnt", 512)
        self.image_size = config.get("image_size", 448)
        self.vit = config.get("vit", 'unitable-large')
        self.vit_grad_ckpt = config.get("vit_grad_ckpt", False)
        self.vit_ckpt_layer = config.get("vit_ckpt_layer", 0)
        self.vit_ckpt_path = config.get("vit_ckpt_path", None)
        self.bert_ckpt_path = config.get("bert_ckpt_path", None)

        vision_width = 768
        self.visual_encoder = Unitable_Encoder(
            img_size=self.image_size, patch_size=16, d_model=vision_width, 
            nlayer=12, nhead=12, dropout=0.1
        )

        if not config.get('pretrained', ''):
            assert self.vit_ckpt_path is not None
            print(f"Unitable model weight path: {self.vit_ckpt_path}")
            checkpoint = torch.load(self.vit_ckpt_path, map_location="cpu")
            msg = self.visual_encoder.load_state_dict(checkpoint, strict=False)
            assert len(msg.missing_keys) == 0, f"{msg.missing_keys}"
            assert msg.unexpected_keys == ["generator.weight", "generator.bias"], f"{msg.unexpected_keys}"
        self.visual_encoder.interpolate_position_embeddings()

        ######## Tokenizer ########
        self.bert_ckpt_path = self.bert_ckpt_path if self.bert_ckpt_path is not None else 'bert-base-uncased'
        assert self.bert_ckpt_path is not None, "测试用"
        self.tokenizer = init_tokenizer_tablip(self.bert_ckpt_path)

        data_ids = ["C-tag"]
        self.data_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in data_ids]

        ######## Text Encoder ########
        encoder_config = BertConfig.from_json_file(med_config)
        assert encoder_config.encoder_width == vision_width 
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False) 
        
        ######## Text Decoder ########
        decoder_config = BertConfig.from_json_file(med_config)
        assert decoder_config.encoder_width == vision_width
        self.text_decoder = BertLMHeadModel(config=decoder_config)
        
        if not config.get('pretrained', ''):
            self.load_bert_state(encoder_config, self.bert_ckpt_path)

        # FIXME
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))



        ######## Bbox emebedding #######
        hidden_size = vision_width
        assert hidden_size % 4 == 0, "hidden_size must be divisible by 4"
        self.x_embedding = nn.Embedding(
            self.image_size + 4,
            hidden_size // 4,                   
            padding_idx=self.image_size + 3,    
        )
        self.y_embedding = nn.Embedding(
            self.image_size + 4,
            hidden_size // 4, 
            padding_idx=self.image_size + 3,
        )

        ######## Layout Pointer ########
        self.bbox_linear = nn.Linear(
            hidden_size, hidden_size, bias=False
        )
        self.otsl_linear = nn.Linear(
            hidden_size, hidden_size, bias=False
        )

        ######## ROIAlignment  ########
        if config.get("use_RoiAlign", False):
            self.img_downsize_scale = 16
            assert (
                self.image_size == 448
            ), "input_size must be (768, 768) when use_imgRoiAlign is True"
            self.roi_align = RoIAlign(
                output_size=(2, 2),
                spatial_scale=1 / self.img_downsize_scale,
                sampling_ratio=-1,
                aligned=False,
            )
            self.roi_proj = nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )

            self.empty_embed = nn.Embedding(1, hidden_size)
            self.bbox_coord_merge = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.roi_merge = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        self.bbox_positions = torch.arange(self.bbox_token_cnt)


    def load_bert_state(self, config, path: str = "bert-base-uncased"):
        state_dict = BertModel.from_pretrained(path).state_dict()
        en_tgt_state_dict = self.text_encoder.state_dict()
        de_tgt_state_dict = self.text_decoder.state_dict()
        en_state_dict = {}
        de_state_dict = {}
        num_layer = config.num_hidden_layers
        for name, parms in state_dict.items():
            if name in en_tgt_state_dict:
                en_state_dict[name] = parms
            if "bert." + name in de_tgt_state_dict:
                de_state_dict["bert." + name] = parms
        en_msk = self.text_encoder.load_state_dict(en_state_dict, strict=False)
        de_msk = self.text_decoder.load_state_dict(de_state_dict, strict=False)

        if not en_msk.missing_keys and not de_msk.missing_keys:
            print("成功：所有目标权重都已从预训练模型中找到并加载。")
        else:
            print(f"警告: 以下权重在Encoder模型中存在，但未被加载 (将被随机初始化): {en_msk.missing_keys}")
            print(f"警告: 以下权重在Docoder模型中存在，但未被加载 (将被随机初始化): {de_msk.missing_keys}")

        print(f"信息: 以下权重在预训练模型中存在，但被忽略了: {en_msk.unexpected_keys}")
        print(f"信息: 以下权重在预训练模型中存在，但被忽略了: {de_msk.unexpected_keys}")


    def inference(
        self: "TaBLIP",
        image: torch.Tensor,
        dr_coords: torch.Tensor,
        valid_coord_lens: torch.Tensor,
        num_beams: int = 1,
    ):  
        B = image.shape[0]
        image_embeds = self.visual_encoder(image)[:, 1:]
        assert image_embeds.shape[1:] == (784, 768)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        assert dr_coords.shape[1] == (self.bbox_token_cnt - 1), \
            f"dr_coords: {dr_coords.shape}, bbox_token_cnt: {self.bbox_token_cnt}"
        
        dr_coords = dr_coords.to(image.device)
        valid_coord_lens = valid_coord_lens.to(image.device)

        layout_output = self.get_layout_output(
            image_embeds, image_atts, dr_coords, valid_coord_lens
        )

        bos_ids = torch.full(
            (B, 1),
            fill_value=self.tokenizer.bos_token_id,
            device=image.device
        )
        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts
        }

        decoder_output = self.text_decoder.generate(
            input_ids=bos_ids,
            max_length=self.max_length,
            min_length=1,
            num_beams=num_beams, 
            do_sample=False,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id, 
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            use_cache=True, 
            return_dict_in_generate=True,
            output_hidden_states=True,
            **model_kwargs
        )
        structure_output = decoder_output.hidden_states[-1][-1]
        text_token_seq = decoder_output.sequences  

        otsl_feature = self.otsl_linear(structure_output)
        bbox_feature = self.bbox_linear(layout_output)

        combined_feature = []
        iter_size = 4
        for i in range(0, B, iter_size):
            norm_otsl_feature = F.normalize(otsl_feature[i : i + iter_size], dim=-1)
            norm_bbox_feature = F.normalize(bbox_feature[i : i + iter_size], dim=-1)
            tmp_feature = torch.bmm(
                norm_otsl_feature, 
                norm_bbox_feature.transpose(1, 2)
            )
            combined_feature.append(tmp_feature)

        combined_feature = torch.cat(combined_feature, dim=0)
        coord_logits = combined_feature[:, :, 1:] 

        text_token_seq = text_token_seq[:, 1:]
        is_data_tensor = torch.zeros(
            (text_token_seq.shape[0], text_token_seq.shape[1]),
            dtype=torch.bool,
            device=text_token_seq.device,
        )  

        for data_id in self.data_ids:
            is_data_tensor = torch.logical_or(
                is_data_tensor, text_token_seq == data_id
            ) 
        coord_logits[~is_data_tensor] = float("-inf")
        coord_one_hot = F.one_hot(
            torch.argmax(coord_logits, dim=1), 
            num_classes=coord_logits.shape[1],
        ).transpose(1, 2)

        is_empty = torch.sum(coord_one_hot, dim=-1) == 0  
        is_empty = is_empty.unsqueeze(-1).to(coord_one_hot.dtype)  

        combined_feature = torch.cat([is_empty, coord_one_hot], dim=-1)  

        return {
            "layout_output": layout_output, 
            "structure_output": structure_output, 
            "text_to_dr_coord": combined_feature,
            "output_sequences":  decoder_output.sequences,
        }


    def get_layout_output(
            self, image_embeds, image_atts, dr_coords, valid_coord_lens):
        B, _, _ = dr_coords.shape
        layout_embedding = self.embed_coord(dr_coords)
        empty_coord = torch.tensor(
            [
                self.image_size + 1,
                self.image_size + 1,
                self.image_size + 2,
                self.image_size + 2,
            ],
            dtype=dr_coords.dtype,
            device=dr_coords.device,
        ).unsqueeze(0).unsqueeze(0) 
        assert empty_coord.shape == (1, 1, 4), empty_coord.shape

        empty_coord_embedding = self.embed_coord(empty_coord)
        empty_coord_embedding = empty_coord_embedding.expand(B, -1, -1)
        layout_embedding = torch.cat(
            (empty_coord_embedding, layout_embedding),
            dim=1
        )
        assert layout_embedding.shape[:2] == (B, self.bbox_token_cnt), layout_embedding.shape

        if self.config.get("use_RoiAlign", False):
            align_feature = self.get_img_roiAlign(image_embeds, dr_coords) 
            empty_align = self.empty_embed.weight.unsqueeze(0).repeat(
                B, 1, 1)  
            align_feature = torch.cat(
                [empty_align, align_feature], dim=1
            ) 
            layout_embedding = self.bbox_coord_merge(layout_embedding) + \
                self.roi_merge(align_feature)

        valid_coord_lens = (valid_coord_lens + 1).unsqueeze(1) # +1 for empty coodr
        layout_attention_mask = self.bbox_positions.to(image_embeds.device) < valid_coord_lens
        assert layout_attention_mask.shape == (B, self.bbox_token_cnt)

        encoder_output = self.text_encoder.forward(
            inputs_embeds = layout_embedding,
            attention_mask = layout_attention_mask,
            encoder_hidden_states = image_embeds,
            encoder_attention_mask = image_atts,
            return_dict = True,
        )
        layout_output = encoder_output.last_hidden_state
        return layout_output


    def embed_coord(self, coord_tensor: torch.Tensor):
        """Embed coordinate tensor"""
        assert coord_tensor.shape[-1] == 4
        coord_embedding = torch.cat(
            [
                self.x_embedding(coord_tensor[..., 0]),
                self.y_embedding(coord_tensor[..., 1]),
                self.x_embedding(coord_tensor[..., 2]),
                self.y_embedding(coord_tensor[..., 3]),
            ],
            dim=-1,
        )
        return coord_embedding


    def get_img_roiAlign(self, image_embeds, dr_coords):
        """
        Get Image ROIAlign based on input coordinates
        """
        org_dtype = image_embeds.dtype
        device = image_embeds.device
        B, num_bbox, _ = dr_coords.shape

        img_idx = torch.arange(B, device=device)[:, None, None] 
        img_idx = img_idx.repeat(1, dr_coords.shape[1], 1)

        rois = torch.cat((img_idx, dr_coords), dim=-1).to(torch.float)
        rois = rois.view(-1, 5)

        H = int(self.image_size / self.img_downsize_scale)
        W = int(self.image_size / self.img_downsize_scale)
        feature_map = image_embeds.transpose(1, 2).view(
            B, image_embeds.shape[-1], H, W
        ).to(torch.float)

        pooled_features = self.roi_align(feature_map, rois)
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)
        pooled_features = self.roi_proj(pooled_features.to(org_dtype))  
        pooled_features = pooled_features.view(B, num_bbox, -1)

        return pooled_features
    


def tablip(pretrained='',**kwargs):
    model = TaBLIP(**kwargs)
    if pretrained:
        model, msg = load_tablip_checkpoint(model, pretrained)
        if not msg.missing_keys:
            print("成功：所有目标权重都已从预训练模型中找到并加载。")
        else:
            print(f"警告: 以下权重在 model 模型中存在，但未被加载 (将被随机初始化): {msg.missing_keys}")
        for key in msg.unexpected_keys:
            print(f"信息: 以下权重在预训练模型中存在，但被忽略了: {key}")

    tie_encoder_decoder_weights(
        model.text_encoder, 
        model.text_decoder.bert, 
        '',
        '/attention'
    )
    return model


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_tablip_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    state_dict['visual_encoder.pos_embed.embedding.weight'] = interpolate_pos_embed(
        state_dict['visual_encoder.pos_embed.embedding.weight'], model.visual_encoder
    )
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed.embedding.weight']=interpolate_pos_embed(
            state_dict['visual_encoder_m.pos_embed.embedding.weight'],
            model.visual_encoder_m
        )    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model, msg


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    
        

def init_tokenizer_tablip(ckptpath):
    special_chars = [
        'rowspan="','colspan="','<td ','<tr>','</tbody>',
        '</td>','<thead>','<tbody>','</tr>','</thead>',
        '<td>','<sep/>','<s_answer>','</s_answer>',
        'C-tag', 'U-tag','L-tag','X-tag','NL-tag','R-tag',
        '<CELL>'
    ]
    tokenizer = BertTokenizer.from_pretrained(ckptpath)
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'eos_token':'[EOS]'})
    all_additional_tokens = ['[ENC]'] + sorted(list(set(special_chars)))
    tokenizer.add_special_tokens(
        {'additional_special_tokens':all_additional_tokens})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer



def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        print(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias                
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)  