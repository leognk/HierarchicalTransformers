import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.models.layers import DropPath
from einops import rearrange
from utils import product, ravel_multi_index


class MLP(torchvision.ops.MLP):

    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__(in_dim, [hidden_dim, in_dim], activation_layer=nn.GELU, dropout=dropout)
    

class Attention(nn.Module):

    def __init__(self, self_attend, emb_dim, n_heads, dropout, use_flash_attn, rel_pos_ctx_size=None):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.self_attend = self_attend
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn
        if not use_flash_attn:
            self.scale = (emb_dim // n_heads) ** -0.5
        if self_attend:
            self.to_qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=True)
        else:
            self.to_q = nn.Linear(emb_dim, emb_dim, bias=True)
            self.to_kv = nn.Linear(emb_dim, 2 * emb_dim, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

        # Prepare relative position bias.
        self.add_rel_pos = rel_pos_ctx_size is not None
        if self.add_rel_pos:
            assert self_attend
            self.tot_ctx_size = product(rel_pos_ctx_size)
            rge = [2 * s - 1 for s in rel_pos_ctx_size]
            self.rel_pos_bias_table = nn.Parameter(torch.zeros(product(rge), n_heads)) # [(r1 ... rd) h], rk = 2*nk-1

            coords = [torch.arange(s) for s in rel_pos_ctx_size]
            coords = torch.stack(torch.meshgrid(coords, indexing='ij')) # [d n1 ... nd]
            coords = torch.flatten(coords, start_dim=1) # [d n]
            rel_coords = coords[:, :, None] - coords[:, None, :] # [d n n]
            rel_coords = torch.flatten(rel_coords, start_dim=1) # [d nn]
            rel_coords += torch.tensor(rel_pos_ctx_size).view(-1, 1) - 1
            rel_coords = ravel_multi_index(rel_coords, rge) # [nn]
            self.register_buffer("rel_pos_ids", rel_coords) # [nn]

        self._initialize_weights()
    
    def _initialize_weights(self):
        if self.add_rel_pos:
            nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)
    
    def forward(self, x, context=None, attn_mask=None):
        """
        Set context to None for self-attention.
        attn_mask: [b n n]
        """
        assert (context is None) == self.self_attend
        if self.self_attend:
            qkv = self.to_qkv(x)
            q, k, v = rearrange(qkv, 'b n (k h d) -> k b h n d', k=3, h=self.n_heads)
        else:
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
            k, v = rearrange(kv, 'b n (k h d) -> k b h n d', k=2, h=self.n_heads)
        if attn_mask is not None: attn_mask = attn_mask.unsqueeze(1) # [b n n] -> [b 1 n n]
        if self.add_rel_pos:
            rel_pos_bias = self.rel_pos_bias_table[self.rel_pos_ids]\
                .view(self.tot_ctx_size, self.tot_ctx_size, self.n_heads)\
                .movedim(-1, 0) # [h n n]
            attn_mask = attn_mask + rel_pos_bias if attn_mask is not None else rel_pos_bias
        if self.use_flash_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout)
        else:
            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            if attn_mask is not None: attn += attn_mask
            attn = torch.nn.functional.softmax(attn, dim=-1)
            x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, self_attend, emb_dim, mlp_hidden_dim, n_heads, dropout, drop_path, use_flash_attn, rel_pos_ctx_size=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.self_attend = self_attend
        if not self_attend:
            self.ctx_norm = nn.LayerNorm(emb_dim)
        self.attn = Attention(self_attend, emb_dim, n_heads, dropout, use_flash_attn, rel_pos_ctx_size)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim, mlp_hidden_dim, dropout)
    
    def forward(self, x, context=None, attn_mask=None):
        """
        Set context to None for self-attention.
        attn_mask: [b n n]
        """
        assert (context is None) == self.self_attend
        if not self.self_attend: context = self.ctx_norm(context)
        x = x + self.drop_path(self.attn(self.norm1(x), context, attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):

    def __init__(self, n_layers, self_attend, emb_dim, mlp_hidden_dim, n_heads, dropout, drop_path, use_flash_attn, rel_pos_ctx_size=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            drop_path_i = drop_path[i] if drop_path else None
            self.blocks.append(TransformerBlock(self_attend, emb_dim, mlp_hidden_dim, n_heads, dropout, drop_path_i, use_flash_attn, rel_pos_ctx_size))
    
    def forward(self, x, context=None, attn_mask=None):
        """
        Set context to None for self-attention.
        attn_mask: [b n n]
        """
        for block in self.blocks:
            x = block(x, context, attn_mask)
        return x