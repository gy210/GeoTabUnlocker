import torch
from torch import nn, Tensor

class ImgLinearBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_size: int,
        in_chan: int = 3,
    ) -> None:
        super().__init__()

        self.conv_proj = nn.Conv2d(
            in_chan, out_channels=d_model, kernel_size=patch_size, stride=patch_size
        )
        self.d_model = d_model
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_proj(x)
        x = x.flatten(start_dim=-2).transpose(1, 2)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.embedding(torch.arange(x.shape[1], device=x.device))
        return self.dropout(out + x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        nlayer: int,
        ff_ratio: int = 4,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=nhead,
            dim_feedforward=ff_ratio * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return x
    

class Unitable_Encoder(nn.Module):
    def __init__(
        self,
        img_size: int = 448,
        patch_size: int = 16,
        d_model: int = 768,
        nhead: int = 12,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        nlayer: int = 12,
        ff_ratio: int = 4,
    ) -> None:
        super().__init__()
        max_seq_len = int(img_size // patch_size) ** 2
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.backbone = ImgLinearBackbone(d_model=d_model, patch_size=patch_size, in_chan=3)

        self.encoder = Encoder(
            d_model, nhead, dropout, activation, norm_first, nlayer, ff_ratio)
        
        self.pos_embed = PositionEmbedding(
            max_seq_len=max_seq_len, d_model=d_model, dropout=dropout
        )
        self.norm = nn.LayerNorm(d_model)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            self.trunc_normal(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            self.trunc_normal(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, PositionEmbedding):
            self.trunc_normal(m.embedding.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "mask_token"}
    
    def forward(self, x: Tensor):
        x = self.backbone(x)
        B, S, E = x.shape
        assert E == self.d_model
        
        cls_tokens = self.mask_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_embed(x)

        x = self.encoder(x)
        x = self.norm(x)

        return x
    

    def interpolate_position_embeddings(self) -> torch.Tensor:
        max_seq_len, d_model = self.pos_embed.embedding.weight.shape

        new_seq_len = self.max_seq_len + 1

        if max_seq_len == new_seq_len:
            return
        
        pos_embed = self.pos_embed.embedding.weight
        
        pos_embed = pos_embed.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)
        pos_embed = torch.nn.functional.interpolate(
            pos_embed,
            size=(new_seq_len, 1), 
            mode='bicubic',        
            align_corners=False
        )

        pos_embed = pos_embed.squeeze(-1).permute(0, 2, 1).squeeze(0)
        assert pos_embed.shape == (max_seq_len + 1, d_model)
        
        self.pos_embed.embedding.weight = nn.Parameter(pos_embed)



def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        

    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.max_seq_len
    num_extra_tokens = visual_encoder.pos_embed.embedding.weight.shape[-2] - num_patches
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint