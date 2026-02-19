# Original code by Khang, Minsoo and Hong, Teakgyu
# Source: https://github.com/UpstageAI/TFLOP/tflop/loss.py
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
from torch import nn
import torch.nn.functional as F


class TableCL(nn.Module):
    """
    A contrastive learning model for table data with masks.

    Attributes:
        temperature (float): Temperature scaling factor for the loss function.
        sup_con_loss (SupConLoss): Instance of the supervised contrastive loss.
    """

    def __init__(self, temperature=0.1):
        """
        Initialize the TableCL module.

        Args:
            temperature (float): Temperature scaling factor for the contrastive loss.
        """
        super(TableCL, self).__init__()
        self.temperature = temperature
        self.sup_con_loss = SupConLoss(temperature)

    def forward(self, features, masks, input_coords_length):
        """
        Compute the batch loss for the given features and masks.

        Args:
            features (torch.Tensor): shape [batch_size, bbox_token_cnt, d_model].
            masks (torch.Tensor): shape [batch_size, num_layers, bbox_token_cnt, bbox_token_cnt].
            input_coords_length (torch.Tensor): Lengths of input coordinates, shape [batch_size].

        Returns:
            torch.Tensor: Average batch loss.
        """
        batch_loss, valid_batch_size = 0, 0
        B, _, bbox_tok_cnt, _ = masks.shape
        assert bbox_tok_cnt == features.shape[1]

        for idx in range(B):
            valid_len = input_coords_length[idx] + 1
            selected_mask = masks[idx][:, :valid_len, :valid_len]  # [1, valid_len, valid_len]
            assert selected_mask.shape[0] == 1, "?????有什么用?????"

            selected_feature = features[idx][:valid_len]  # [valid_len, d_model]
            selected_feature = selected_feature.unsqueeze(0)  # [1, valid_len, d_model]

            batch_loss += self.sup_con_loss(selected_feature, mask=selected_mask)

            # check if the data is valid
            float_selected_mask = selected_mask[0].to(torch.float)  # [valid_len, valid_len]
            sanity_tensor = torch.eye(
                float_selected_mask.shape[0],
                dtype=float_selected_mask.dtype,
                device=float_selected_mask.device,
            )
            sanity_tensor[0, 0] = 0
            if torch.sum(float_selected_mask != sanity_tensor) != 0:
                valid_batch_size += 1

        valid_batch_size = max(valid_batch_size, 1)
        batch_loss = batch_loss / valid_batch_size

        return batch_loss
    


class SupConLoss(nn.Module):
    """
    A PyTorch implementation of a modified version of Supervised Contrastive Loss.

    Args:
        temperature (float): Temperature scaling factor for contrastive loss. Default is 0.1.

    Methods:
        forward(features, mask):
            Computes the modified supervised contrastive loss for the given features and mask.
    """

    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, mask):
        """
        Forward pass to compute the supervised contrastive loss.

        Args:
            features (torch.Tensor):shape [batch_size, bbox_token_cnt, hidden_size].
            masks (torch.Tensor): shape [batch_size, num_layers, bbox_token_cnt, bbox_token_cnt].

        Returns:
            torch.Tensor: A scalar tensor representing the computed contrastive loss.
        """
        B, bbox_token_cnt, hidden_size = features.shape

        dot_contrast = torch.div(
            torch.matmul(features, features.transpose(1, 2)), self.temperature
        )

        logits_max, _ = torch.max(
            dot_contrast, dim=-1, keepdim=True
        ) 
        logits = (
            dot_contrast - logits_max.detach()
        )
        logits_mask = 1 - torch.eye(
            bbox_token_cnt, dtype=logits.dtype, device=features.device
        ).unsqueeze(
            0
        )  
        exp_logits = (torch.exp(logits) * logits_mask)

        negative_mask = 1 - mask
        negative_mask[negative_mask < 1] = 0
        negative_denom = torch.sum(
            exp_logits * negative_mask, dim=-1, keepdim=True
        ) 

        positive_mask = mask.clone()
        positive_mask[positive_mask > 0] = 1
        positive_denom = torch.sum(
            exp_logits * positive_mask, dim=-1, keepdim=True
        ) 

        denominator = negative_denom + positive_denom.detach() + 1e-6
        log_prob = logits - torch.log(denominator)

        mask = mask * logits_mask
        mean_log_prob_pos = (mask * log_prob).sum(-1) / (
            mask.sum(-1) + 1e-6
        ) 

        loss = -1 * mean_log_prob_pos.mean()

        return loss