from models.unitable_encoder_large import Unitable_Encoder, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, load_checkpoint
from models.tablip_pretrain import tie_encoder_decoder_weights
from models.loss import TableCL

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


class TaBLIP_posttrain(nn.Module):
    def __init__(self, config, med_config = 'configs/med_config.json'):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()

        self.config = config
        self.max_length = config.get("max_length", 256)
        self.bbox_token_cnt = config.get("bbox_token_cnt", 512)
        self.image_size = config.get("image_size", 448)
        self.vit = config.get("vit", 'unitable-large')
        self.vit_grad_ckpt = config.get("vit_grad_ckpt", False)
        self.vit_ckpt_layer = config.get("vit_ckpt_layer", 0)
        self.vit_ckpt_path = config.get("vit_ckpt_path", None)
        self.bert_ckpt_path = config.get("bert_ckpt_path", None)
        self.weight_loss_lm = config.get("weight_loss_lm", 1.0)
        self.weight_loss_bce = config.get("weight_loss_bce", 1.0)
        self.weight_loss_pointer = config.get("weight_loss_pointer", 1.0)
        self.weight_loss_Span = config.get("weight_loss_Span", 0.5)
        self.use_bbox_HiMulConET = config.get("use_bbox_HiMulConET", True)
        self.contrastive_loss_config = {
            "use_RowWise_contLearning": config.get("use_RowWise_contLearning", False),
            "use_ColWise_contLearning": config.get("use_ColWise_contLearning", False),
        }
        self.span_coeff_mode = self.config.get("span_coeff_mode", "proportional")
        self.use_dist = config.get("use_dist", False)
        self.sigma = config.get("sigma", 1.0)

        ######## Visual Encoder ########
        self.visual_encoder, vision_width = create_vit(
            self.vit, self.image_size, self.vit_grad_ckpt, self.vit_ckpt_layer, drop_path_rate=0.1)
        
        if not config.get('pretrained', ''): # pretrain 模型不存在时，加载参数
            assert self.vit_ckpt_path is not None
            print(f"Unitable 模型路径: {self.vit_ckpt_path}")
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

        ######## row-wise and column-wise ########
        if config.get("use_bbox_HiMulConET", False):
            # Set up modules for row-wise and column-wise linear transformation
            self.rowwise_linear = nn.Linear(hidden_size, hidden_size, bias=False)
            self.colwise_linear = nn.Linear(hidden_size, hidden_size, bias=False)
            self.TableCL_loss = TableCL(temperature=0.1)

        ######## ROIAlignment ########
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
        print(f"Bert模型路径: {path}")
        en_tgt_state_dict = self.text_encoder.state_dict()
        de_tgt_state_dict = self.text_decoder.state_dict()
        en_state_dict = {}
        de_state_dict = {}
        num_layer = config.num_hidden_layers
        print(f"开始筛选前 {num_layer} 层的权重...")
        for name, parms in state_dict.items():
            if name in en_tgt_state_dict:
                en_state_dict[name] = parms
            if "bert." + name in de_tgt_state_dict:
                de_state_dict["bert." + name] = parms
        print("正在将筛选后的权重加载到模型中...")
        en_msk = self.text_encoder.load_state_dict(en_state_dict, strict=False)
        de_msk = self.text_decoder.load_state_dict(de_state_dict, strict=False)

        print("\n--- 权重加载报告 ---")
        if not en_msk.missing_keys and not de_msk.missing_keys:
            print("成功：所有目标权重都已从预训练模型中找到并加载。")
        else:
            print(f"警告: 以下权重在Encoder模型中存在，但未被加载 (将被随机初始化): {en_msk.missing_keys}")
            print(f"警告: 以下权重在Docoder模型中存在，但未被加载 (将被随机初始化): {de_msk.missing_keys}")

        print(f"信息: 以下权重在预训练模型中存在，但被忽略了: {en_msk.unexpected_keys}")
        print(f"信息: 以下权重在预训练模型中存在，但被忽略了: {de_msk.unexpected_keys}")
        print()


    def forward(
        self: "TaBLIP_posttrain", 
        image: torch.Tensor,              
        otsl_seq_list: List[str],
        dr_coords: torch.Tensor,          
        valid_coord_lens: torch.Tensor,   
        pointer_labels: torch.Tensor,     
        pointer_mask_labels: torch.Tensor,
        bbox_coeff_tensors: torch.Tensor, 
        ) -> torch.Tensor:
        """
        Args:
            image: Tensor(B, P, hidden_size)
            otsl_seq_list: List of OTSL sequence
            dr_coords: Tensor(B, bbox_token_cnt - 1, 4)
            valid_coord_lens: Tensor(B,)
            pointer_labels: Tensor(B, len(otsl_seq), bbox_token_cnt)
            pointer_mask_labels: Tensor(B, len(otsl_seq))
            bbox_coeff_tensors: Tensor(B, 2, bbox_token_cnt, bbox_token_cnt)
            mode: 'ocr' or 'detect'. ocr: 直接对应; detect: 检测文本的bbox并对应
        """
        B = image.shape[0]
        loss = None    

        image_embeds = self.visual_encoder(image)[:, 1:] 
        assert image_embeds.shape[1:] == (784, 768)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        assert dr_coords.shape[1] == (self.bbox_token_cnt - 1), \
            f"dr_coords: {dr_coords.shape}, bbox_token_cnt: {self.bbox_token_cnt}"
        
        valid_coord_lens = valid_coord_lens.to(image.device)
        pointer_labels = pointer_labels.to(image.device)
        pointer_mask_labels = pointer_mask_labels.to(image.device)
        bbox_coeff_tensors = bbox_coeff_tensors.to(image.device)
        dr_coords = dr_coords.to(image.device)

        layout_output = self.get_layout_output(
            image_embeds, image_atts, dr_coords, valid_coord_lens
        )

        otsl = self.tokenizer(otsl_seq_list, padding='longest', truncation=True, max_length=self.max_length, 
                                return_tensors="pt").to(image.device)
        assert otsl.input_ids.shape[1] <= self.max_length
        otsl.input_ids[:,0] = self.tokenizer.bos_token_id
        otsl_targets = otsl.input_ids.masked_fill(
            otsl.input_ids == self.tokenizer.pad_token_id, -100)
                        
        decoder_output = self.text_decoder.forward(
            input_ids = otsl.input_ids, 
            attention_mask = otsl.attention_mask, 
            encoder_hidden_states = image_embeds,
            encoder_attention_mask = image_atts,                  
            labels = otsl_targets,
            return_dict = True, 
            output_hidden_states=True,
        )
        structure_output = decoder_output.hidden_states[-1] # FIXME: 
        loss_lm = decoder_output.loss

        bbox_coeff_tensors = self.bbox_dist_weight(
            dr_coords, bbox_coeff_tensors, self.sigma
        )


        loss_ptr = 0
        pointer_acc = 0
        bbox_TableCL_loss = 0
        (loss_rowwise, loss_colwise) = (0, 0)
        if pointer_labels.shape[1] > (otsl.input_ids.shape[1] - 2):
            L = self.max_length - 2
            pointer_labels = pointer_labels[:, : L, :]
            pointer_mask_labels = pointer_mask_labels[:, : L]
        
        B_sub = 8
        num_B_sub = B // B_sub
        if B % B_sub > 0:
            num_B_sub += 1
        for idx in range(num_B_sub):
            S = slice(idx * B_sub, (idx + 1) * B_sub)
            loss_bce, loss_pointer, ptr_acc = self.get_tag2coord_ptr_loss(
                layout_output = layout_output[S],
                structure_output = structure_output[S],
                valid_coord_lens = valid_coord_lens[S],
                pointer_labels = pointer_labels[S],
                pointer_mask_labels = pointer_mask_labels[S],
            )

            curr_B = B_sub
            if (B % B_sub > 0) and (idx == num_B_sub - 1):
                curr_B = B % B_sub

            loss_ptr += (
                (self.weight_loss_bce * loss_bce) + 
                (self.weight_loss_pointer * loss_pointer)
            ) * curr_B
            pointer_acc += ptr_acc * curr_B
            
            if self.use_bbox_HiMulConET:
                curr_rowwise_loss, curr_colwise_loss = self.get_bbox_TableCL_loss(
                    bbox_coeff_tensor = bbox_coeff_tensors[S],
                    layout_output = layout_output[S],
                    input_coords_length = valid_coord_lens[S],
                    contr_learning_config = self.contrastive_loss_config,
                )

                curr_bbox_TableCL_loss = (curr_rowwise_loss + curr_colwise_loss) * curr_B
                curr_bbox_TableCL_loss /= sum(self.contrastive_loss_config.values())
                bbox_TableCL_loss += curr_bbox_TableCL_loss

                loss_rowwise += curr_rowwise_loss * curr_B
                loss_colwise += curr_colwise_loss * curr_B
            
        loss_ptr /= B
        pointer_acc /= B
        if loss is None:
            loss = self.weight_loss_lm * loss_lm + loss_ptr
        else:
            loss += self.weight_loss_lm * loss_lm + loss_ptr
        
        if self.use_bbox_HiMulConET:
            loss_rowwise /= B  
            loss_colwise /= B  
            bbox_TableCL_loss /= B
            if loss is None:
                loss = self.weight_loss_Span * bbox_TableCL_loss
            else:
                loss += self.weight_loss_Span * bbox_TableCL_loss
        
        return {
            "loss": loss,
            "loss_lm": loss_lm,
            "loss_ptr": loss_ptr,
            "loss_rowwise": loss_rowwise,
            "loss_colwise": loss_colwise,
            "pointer_acc": pointer_acc,
        }
                  

    def inference(
        self: "TaBLIP_posttrain",
        image: torch.Tensor,
        dr_coords: torch.Tensor,
        valid_coord_lens: torch.Tensor,
        num_beams: int = 1,
    ):  
        self.eval()
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


    def get_tag2coord_ptr_loss(
        self,
        layout_output,
        structure_output,
        valid_coord_lens,
        pointer_labels,
        pointer_mask_labels,
    ):        
        '''
            structure_output: shape(B, L + 2, bbox_token_cnt)
            pointer_labels: shape(B, L, bbox_token_cnt), L is the num of uesful token(except [DEC] and [SEP])
        '''
        assert layout_output.shape[1] == self.bbox_token_cnt, \
            "output_seq must be (B, bbox_token_cnt, hidden_size)"
        assert structure_output.shape[1] <= self.max_length

        B = layout_output.shape[0]

        otsl_feature = self.otsl_linear(structure_output[:, :-2, :]) 
        bbox_feature = self.bbox_linear(layout_output)
        otsl_feature = F.normalize(otsl_feature, dim=-1)
        bbox_feature = F.normalize(bbox_feature, dim=-1)
        
        combined_feat = torch.bmm(
            otsl_feature, bbox_feature.transpose(1, 2)
        ) 
        if pointer_labels.dtype != otsl_feature.dtype:
            pointer_labels = pointer_labels.to(otsl_feature.dtype)
        
        temperature = 0.1
        loss_bce = 0
        loss_pointer = 0
        
        batchwise_pointing_acc = []
        for data_i in range(B):
            is_data_only_pred = combined_feat[data_i, pointer_mask_labels[data_i]]    
            is_data_only_label = pointer_labels[data_i, pointer_mask_labels[data_i]]  

            loss_bce += nn.BCEWithLogitsLoss()(
                is_data_only_pred[:, 0], is_data_only_label[:, 0]
            )

            L_bbox = valid_coord_lens[data_i]

            is_not_empty_pred = is_data_only_pred[:, 1 : (L_bbox + 1)] 

            is_not_empty_label = is_data_only_label[:, 1 : (L_bbox + 1)]
            
            valid_coords_tmp = (torch.sum(is_not_empty_label, 0) == 1)  

            is_not_empty_pred = is_not_empty_pred / temperature
            loss_pointer += nn.CrossEntropyLoss()(
                torch.transpose(is_not_empty_pred, 0, 1)[valid_coords_tmp],
                torch.argmax(
                    torch.transpose(is_not_empty_label, 0, 1)[valid_coords_tmp],
                    dim=-1,
                ),
            )
            with torch.no_grad():
                pointing_pred = F.one_hot(
                    torch.argmax(is_not_empty_pred, dim=0),
                    num_classes=is_not_empty_pred.shape[0],
                ).transpose(0, 1) 
                pointing_pred = pointing_pred[:, valid_coords_tmp]

                pointing_label = is_not_empty_label
                
                pointing_label = pointing_label[:, valid_coords_tmp]

                equiv_tns = (pointing_pred == pointing_label) 

                token_wise_equivalence = torch.sum(equiv_tns, dim=-1) == torch.sum(
                    valid_coords_tmp
                ) 
                batchwise_pointing_acc.append(
                    torch.sum(token_wise_equivalence).float()
                    / token_wise_equivalence.shape[0]
                )

        loss_pointer = loss_pointer / B
        loss_bce = loss_bce / B
        ptr_acc = torch.mean(torch.stack(batchwise_pointing_acc, dim=0))

        return loss_bce, loss_pointer, ptr_acc


    def bbox_dist_weight(
        self, bboxes: torch.Tensor, bbox_coeff_tensors, sigma: float = 1.0
    ) -> torch.Tensor:
        """
        Calculates a pairwise weight matrix for a batch of bounding boxes
        using a Gaussian kernel (RBF kernel).
        """
        c_x = (bboxes[..., 0] + bboxes[..., 2]) / 2
        c_y = (bboxes[..., 1] + bboxes[..., 3]) / 2
        centers = torch.stack([c_x, c_y], dim=2)
        
        diff = centers.unsqueeze(2) - centers.unsqueeze(1)
        diag = torch.sqrt(torch.tensor(float(2 * self.image_size**2))).to(diff.device)
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        dist_matrix = (dist_matrix / diag)
        
        dist_weights = torch.exp(-dist_matrix / (2 * sigma ** 2))
        dist_weights = dist_weights.unsqueeze(1).expand(
            -1, bbox_coeff_tensors.size(1), -1, -1,
        )

        bbox_coeff_tensors[:, :, 1:, 1:] *= dist_weights 
        return bbox_coeff_tensors



    def get_bbox_TableCL_loss(
        self,
        bbox_coeff_tensor,
        layout_output,
        input_coords_length,
        contr_learning_config,
    ):
        """Function to calculate Contrastive Learning loss for bbox tokens

        Args:
            bbox_coeff_tensor: (batch_size, 5, bbox_token_length, bbox_token_length)
            layout_output: (batch_size, bbox_token_cnt, d_model)
            input_coords_length: number of bbox tokens in the input sequence
            contr_learning_config: configuration for contrastive learning
        """
        (rowwise_loss, colwise_loss) = (0, 0)

        if contr_learning_config["use_RowWise_contLearning"]:
            rowwise_feature = self.rowwise_linear(layout_output)
            rowwise_feature = F.normalize(rowwise_feature, dim=-1)
            coeff_idx = sum([contr_learning_config["use_RowWise_contLearning"]])- 1

            rowwise_mask = bbox_coeff_tensor[:, coeff_idx : (coeff_idx + 1)]
            rowwise_loss = self.TableCL_loss.forward(
                features=rowwise_feature,
                masks=rowwise_mask,
                input_coords_length=input_coords_length,
            )

        if contr_learning_config["use_ColWise_contLearning"]:
            colwise_feature = self.colwise_linear(layout_output)
            colwise_feature = F.normalize(colwise_feature, dim=-1)
            coeff_idx = (sum([
                            contr_learning_config["use_RowWise_contLearning"],
                            contr_learning_config["use_ColWise_contLearning"],
                        ]) - 1
                    )
            colwise_mask = bbox_coeff_tensor[:, coeff_idx : (coeff_idx + 1)]
            colwise_loss = self.TableCL_loss(
                features=colwise_feature,
                masks=colwise_mask,
                input_coords_length=input_coords_length,
            )

        return rowwise_loss, colwise_loss




def tablip_posttrain(pretrained='',**kwargs):
    model = TaBLIP_posttrain(**kwargs)
    if pretrained:
        model, msg = load_tablip_checkpoint(model, pretrained)
        print("\n--- 权重加载报告 ---")
        if not msg.missing_keys:
            print("成功：所有目标权重都已从预训练模型中找到并加载。")
        else:
            print(f"警告: 以下权重在 model 模型中存在，但未被加载 (将被随机初始化): {msg.missing_keys}")
        for key in msg.unexpected_keys:
            if key.startswith('visual_encoder_m'): continue
            elif key.startswith('text_encoder_m'): continue
            elif key.startswith('vision_proj_m'): continue
            elif key.startswith('text_proj_m'): continue
            else: print(f"信息: 以下权重在预训练模型中存在，但被忽略了: {key}")

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
    return model,msg


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