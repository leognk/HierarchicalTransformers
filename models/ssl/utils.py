import torch
import utils
from einops import pack, unpack
from ..utils import rearrange_nd, to_torch_pad, do_pad
from ..stems import get_patchify


class RandomMasking(torch.nn.Module):
    """
    Args:
        - size (sequence[int]): [n1 ... nd]
        - mask_ratio (float): 0 <= r <= 1
        - group_size (sequence[int], optional): [g1 ... gd]
            default: size
        - mask_size (sequence[int], optional): [m1 ... md]
            default: [1 ... 1]
    """

    def __init__(self, size, mask_ratio, group_size=None, mask_size=None):
        super().__init__()

        self.rearrange_ids = group_size is not None or mask_size is not None
        self.n_tokens = utils.product(size)

        if group_size is None: group_size = size
        assert all(g <= s for s, g in zip(size, group_size))
        gsize = utils.ceil_div_it(size, group_size)
        self.n_groups = utils.product(gsize)

        if mask_size is None: mask_size = [1] * len(size)
        assert all(g % m == 0 for g, m in zip(group_size, mask_size))
        msize = utils.ceil_div_it(group_size, mask_size)
        self.blocks_per_group = utils.product(msize)
        self.tokens_per_block = utils.product(mask_size)

        masked_blocks_per_group = int(mask_ratio * self.blocks_per_group)
        visible_blocks_per_group = self.blocks_per_group - masked_blocks_per_group
        self.n_visible_per_group = visible_blocks_per_group * self.tokens_per_block
        self.n_masked = int(mask_ratio * self.n_tokens)
        self.n_visible = self.n_tokens - self.n_masked
        
        if self.rearrange_ids:
            ids = torch.arange(self.n_tokens).view(*size)
            size_up = utils.mult_it(gsize, utils.mult_it(msize, mask_size))
            pd = to_torch_pad((True, utils.subtract_it(size_up, size)))
            ids = do_pad(ids, pd, value=-1)
            ids = rearrange_nd(ids, '[(k0 g0 m0)] -> ([k0]) ([g0]) ([m0])', {'g': msize, 'm': mask_size})
            self.register_buffer("ids", ids)
    
    def _gather_token_ids(self, block_ids):
        """
        Gather token ids contained in given block ids.
        block_ids: [b k g] -> [b (k g m)], k=n_groups, g=blocks_per_group, m=tokens_per_block
        """
        b, k = block_ids.shape[:2]
        if not self.rearrange_ids:
            return block_ids.reshape(b, -1)
        
        block_ids = torch.gather(
            self.ids.expand(b, -1, -1, -1),
            dim=2,
            index=block_ids.unsqueeze(-1).expand(-1, -1, -1, self.tokens_per_block),
        ).view(b, k, -1)

        block_ids = torch.cat([
            block_ids[:, :, :self.n_visible_per_group].reshape(b, -1),
            block_ids[:, :, self.n_visible_per_group:].reshape(b, -1),
        ], dim=1)
        block_ids = block_ids[block_ids != -1].reshape(b, -1)
        return block_ids
    
    def remove_tokens(self, x):
        """
        Mask x by removing some of its tokens.
        x: [b n1 ... nd d] or [b n d] -> [b n' d]
        masks: [b n], 1 is masked, 0 is visible
        """
        b, d = x.shape[0], x.shape[-1]
        noise = torch.rand(b, self.n_groups, self.blocks_per_group, device=x.device)
        shuffle_ids = torch.argsort(noise, dim=-1)
        shuffle_ids = self._gather_token_ids(shuffle_ids)
        self.unshuffle_ids = torch.argsort(shuffle_ids, dim=-1)
        visible_ids = shuffle_ids[:, :self.n_visible]

        masks = torch.ones(b, self.n_tokens, device=x.device)
        masks[:, :self.n_visible] = 0
        masks = torch.gather(masks, dim=1, index=self.unshuffle_ids)

        # Mask x
        x, self.ps = pack([x], 'b * d')
        x = torch.gather(x, dim=1, index=visible_ids.unsqueeze(-1).repeat(1, 1, d))

        return x, masks
    
    def add_mask_tokens(self, x, mask_token):
        """
        Fill in x with mask tokens to replace previously removed tokens.
        x: [b n' d] -> [b n1 ... nd d] or [b n d]
        mask_token: [d]
        """
        b, d = x.shape[0], x.shape[-1]
        mask_tokens = mask_token.view(1, 1, -1).repeat(b, self.n_masked, 1)

        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=self.unshuffle_ids.unsqueeze(-1).repeat(1, 1, d))
        [x] = unpack(x, self.ps, 'b * d')

        return x
    
    def replace_with_mask_tokens(self, x, mask_token):
        """
        Mask x by replacing some of its tokens with the mask token.
        x: [b n1 ... nd d] -> [b n1 ... nd d]
        mask_token: [d]
        """
        x, masks = self.remove_tokens(x)
        x = self.add_mask_tokens(x, mask_token)
        return x, masks


class MAELoss(torch.nn.Module):

    def __init__(self, patch_size=None):
        super().__init__()
        self.patchify = get_patchify(patch_size) if patch_size is not None else None
    
    def forward(self, x, tgts, masks, norm_tokens, patchify=False):
        """
        x: [b c N1 ... Nd] if patchify else [b n1 ... nd d]
        tgts: [b c N1 ... Nd] if patchify else [b n1 ... nd d]
        masks: [b n]
        """
        # Reshape
        if patchify:
            assert self.patchify is not None
            x, _ = self.patchify(x) # [b c N1 ... Nd] -> [b n1 ... nd d]
            tgts, _ = self.patchify(tgts) # [b c N1 ... Nd] -> [b n1 ... nd d]
        b, d = x.shape[0], x.shape[-1]
        x = x.view(b, -1, d) # -> [b n d]
        tgts = tgts.view(b, -1, d) # -> [b n d]

        if norm_tokens:
            mean = tgts.mean(dim=-1, keepdim=True)
            var = tgts.var(dim=-1, keepdim=True)
            tgts = (tgts - mean) / (var + 1e-5).sqrt()

        loss = (x - tgts).square().mean(dim=-1) # [b n], loss per token
        loss = (loss * masks).sum() / masks.sum() # loss over masked tokens
        return loss