import math
from collections import defaultdict
from ...utils import *
import utils


class SizeInfo:
    """Information on size for one level."""
    
    def __init__(self):
        self.real_size = None
        self.size = None
        self.pad = None
        self.padded_size = None
        self.n_groups = None
        self.group_size = None
        self.scale_factor = None


class SizeInfosFormatter:
    """Format the information on sizes for each level into a pretty string."""

    def __init__(self):
        self.stage_titles = {
            1: 'FORWARD',
            -1: 'BACKWARD',
        }
        self.heads = {
            'position': 'lvl',
            'scale_factor': 'scale',
            'size': 'size',
            'real_size': 'real',
            'pad': 'pad',
            'padded_size': 'psize',
            'n_groups': 'ngrp',
            'group_size': 'gsize',
        }
        self.none_str = '.'
        self.n_decimals = {'real_size': 2, 'scale_factor': 2}

        self.elms_str = {1: [], -1: []}
        self.elms_max = {1: defaultdict(lambda: None), -1: defaultdict(lambda: None)}
    
    def _get_n_decimals(self, k):
        if k not in self.n_decimals: return None
        return self.n_decimals[k]
    
    @staticmethod
    def _round_str(x, n_decimals):
        if x == int(x): return str(int(x))
        return str(round(x, n_decimals))
    
    @staticmethod
    def _get_elms_str(v, n_decimals=None):
        if v is None: return None
        if n_decimals is None: [str(e) for e in v]
        return [SizeInfosFormatter._round_str(e, n_decimals) for e in v]
    
    @staticmethod
    def _get_elms_len(v):
        if v is None: return None
        return [len(e) for e in v]
    
    @staticmethod
    def _get_elms_max(v1, v2):
        if v1 is None: return v2
        if v2 is None: return v1
        return [max(e1, e2) for e1, e2 in zip(v1, v2)]
    
    def append(self, step, position, s):
        self.elms_str[step].append({})
        es_str = self.elms_str[step][-1]
        es_max = self.elms_max[step]
        es_str['position'] = str(position)
        es_max['position'] = None
        for k, v in vars(s).items():
            es_str[k] = self._get_elms_str(v, self._get_n_decimals(k))
            es_max[k] = self._get_elms_max(es_max[k], self._get_elms_len(es_str[k]))
    
    def _get_vals_str(self, v, m):
        if v is None: return self.none_str
        if isinstance(v, str): return v
        v2 = [e.rjust(n) for e, n in zip(v, m)]
        return f"{' '.join(v2)}"
    
    def get_str(self):
        vals_str = {1: [self.heads.copy()], -1: [self.heads.copy()]}
        heads_len = {k: len(v) for k, v in self.heads.items()}
        vals_max = {1: heads_len.copy(), -1: heads_len.copy()}

        # Calculate vals_str and vals_max.
        n_lvls = len(self.elms_str[1])
        for step in [1, -1]:
            es_max = self.elms_max[step]
            vs_max = vals_max[step]
            for i in range(n_lvls):
                vals_str[step].append({})
                es_str = self.elms_str[step][i]
                vs_str = vals_str[step][-1]
                for k, v in es_str.items():
                    vs_str[k] = self._get_vals_str(v, es_max[k])
                    vs_max[k] = max(vs_max[k], len(vs_str[k]))
            for i in range(len(vals_str[step])):
                vs_str = vals_str[step][i]
                for k in vs_str.keys():
                    vs_str[k] = vs_str[k].rjust(vs_max[k])

        # Build the string.
        stages = []
        for step in [1, -1]:
            lines = [self.stage_titles[step]]
            for i in range(len(vals_str[step])):
                v = vals_str[step][i]
                lines.append(
                    f"{v['position']}: x {v['scale_factor']} = {v['real_size']} ~ {v['size']} + {v['pad']} = {v['padded_size']} = {v['n_groups']} x {v['group_size']}"
                )
            stages.append('\n'.join(lines))
        res = '\n\n'.join(stages)
        return res


class SFTDynamics:
    """Calculate the sizes involved in the SFT model."""

    def __init__(self, in_size, ctx_sizes, qry_sizes):
        self.in_size = in_size
        assert len(ctx_sizes) == len(qry_sizes)
        self.ctx_sizes = ctx_sizes
        self.qry_sizes = qry_sizes
        self.recurrent = len(ctx_sizes) == 1
    
    def _get_group_sizes(self, step):
        if step == 1: return self.ctx_sizes, self.qry_sizes
        elif step == -1: return self.qry_sizes, self.ctx_sizes
    
    def _get_group_size(self, step, i):
        ctx_sizes, qry_sizes = self._get_group_sizes(step)
        if self.recurrent: i = 0
        return ctx_sizes[i], qry_sizes[i]
    
    def get_levels_sizes(self, position1, position2):
        """Return informations on size for each position."""
        if position1 > position2: position1, position2 = position2, position1
        assert position1 <= 0 and 0 <= position2
        if not self.recurrent:
            assert position1 == 0 and position2 <= len(self.ctx_sizes)
        lvls_size = {1: TwoWayList(SizeInfo), -1: TwoWayList(SizeInfo)}
        lvls_size[1][0].real_size = lvls_size[-1][0].real_size = self.in_size
        lvls_size[1][0].size = lvls_size[-1][0].size = self.in_size
        for stop_position in [position1, position2]:
            step = 1 if 0 < stop_position else -1
            s, s1 = lvls_size[step], lvls_size[-step]
            for i in range(0, stop_position, step):
                i1 = i + step
                ctx_size, qry_size = self._get_group_size(step, i)
                need_pad, s[i].pad = get_real_missing_pad(s[i].size, s[i].real_size, ctx_size)
                s[i].padded_size = utils.add_it(s[i].size, s[i].pad)
                s[i].n_groups = utils.floor_div_it(s[i].padded_size, ctx_size)
                s[i].group_size = ctx_size
                s[i1].scale_factor = utils.div_it(qry_size, ctx_size)
                if not need_pad: s[i].pad = s[i].padded_size = None
                s[i1].real_size = utils.mult_it(s[i1].scale_factor, s[i].real_size)
                s[i1].size = utils.mult_it(s[i].n_groups, qry_size)
                s1[i1].real_size = s[i1].real_size
                s1[i1].size = s[i1].size
                s1[i1].n_groups = s[i].n_groups
                s1[i1].group_size = qry_size
                s1[i].scale_factor = utils.div_it(ctx_size, qry_size)
        return lvls_size
    
    def get_levels_sizes_str(self, position1, position2):
        lvls_size = self.get_levels_sizes(position1, position2)
        fmt = SizeInfosFormatter()
        for step in [1, -1]:
            s = lvls_size[step]
            for i in range(s.start_idx, s.stop_idx):
                fmt.append(step, i, s[i])
        return fmt.get_str()
    
    def get_positions(self, target_size):
        """
        For each axis, return the minimum stop position to reach at least the target size.
        Only for recurrent group size.
        """
        assert self.recurrent
        assert len(target_size) == len(self.in_size)
        d = len(self.in_size)
        pos = [None] * d
        for i in range(d):
            s, t = self.in_size[i], target_size[i]
            c, q = self.ctx_sizes[0][i], self.qry_sizes[0][i]
            if s == t:
                pos[i] = 0
                continue
            if c == q:
                continue
            ls, lt, lc, lq = math.log(s), math.log(t), math.log(c), math.log(q)
            # Find the smallest integer k verifying: s*(q/c)**k <= t (when t < s and q < c)
            pos[i] = ceil_div_abs(ls - lt, lc - lq)
        return tuple(pos)
    
    def get_bottleneck_positions(self):
        """
        For each axis, return the minimum stop position to reach the bottleneck.
        Only for recurrent group size.
        """
        grp_size = utils.max_it(self.ctx_sizes[0], self.qry_sizes[0])
        grp_size = utils.min_it(grp_size, self.in_size)
        pos = self.get_positions(grp_size)
        # Cannot use sign(pos) when pos = 0.
        diff = utils.subtract_it(self.ctx_sizes[0], self.qry_sizes[0])
        sgn = [utils.sign(d) for d in diff]
        return tuple(s * (abs(p) + 1) for s, p in zip(sgn, pos))
    
    @staticmethod
    def _max_position(pos):
        a, b = min(pos), max(pos)
        if 0 <= a: return b
        if b <= 0: return a
        return None
    
    def get_position(self, target_size):
        """
        Return the minimum stop position to reach at least the target size.
        Only for recurrent group size.
        """
        pos = self.get_positions(target_size)
        return self._max_position(pos)
    
    def get_bottleneck_position(self):
        """
        Return the minimum stop position to reach the bottleneck.
        Only for recurrent group size.
        """
        pos = self.get_bottleneck_positions()
        return self._max_position(pos)


class PatchifyDynamics:
    """Calculate the sizes involved in the patchify process."""

    def __init__(self, in_size, patch_size):
        assert len(in_size) == len(patch_size)
        self.title = 'PATCHIFY'
        self.heads = {
            'size': 'size',
            'pad': 'pad',
            'padded_size': 'psize',
            'n_patches': 'npatch',
            'patch_size': 'patch',
        }
        self.none_str = '.'
        self.sizes = self._get_sizes(in_size, patch_size)
    
    def _get_sizes(self, in_size, patch_size):
        d = len(in_size)
        sdy = SFTDynamics(in_size, [patch_size], [(1,) * d])
        s = sdy.get_levels_sizes(0, 1)[1][0]
        return {
            'size': s.size,
            'pad': s.pad,
            'padded_size': s.padded_size,
            'n_patches': s.n_groups,
            'patch_size': s.group_size,
        }
    
    def _get_val_str(self, v):
        if v is None: return self.none_str
        v2 = [str(e) for e in v]
        return ' '.join(v2)
    
    def get_sizes_str(self):
        vals_str = [
            self.heads.copy(),
            {k: self._get_val_str(v) for k, v in self.sizes.items()}
        ]
        vals_max = {k: max(len(vals_str[0][k]), len(vals_str[1][k])) for k in self.heads}
        vals_str = [
            {k: v.rjust(vals_max[k]) for k, v in vs_str.items()} for vs_str in vals_str
        ]
        lines = [self.title]
        for v in vals_str:
            lines.append(
                f"{v['size']} + {v['pad']} = {v['padded_size']} = {v['n_patches']} x {v['patch_size']}"
            )
        res = '\n'.join(lines)
        return res


class PipelineDynamics:
    """Calculate the sizes involved in the patchify process and the SFT model."""

    def __init__(self, in_size, patch_size, ctx_sizes, qry_sizes):
        self.pdy = PatchifyDynamics(in_size, patch_size)
        self.sdy = SFTDynamics(self.pdy.sizes['n_patches'], ctx_sizes, qry_sizes)
        self.recurrent = self.sdy.recurrent
    
    def get_sizes(self, position1, position2):
        return [s.size for s in self.sdy.get_levels_sizes(position1, position2)[1]]
    
    def get_sizes_str(self, position1, position2):
        p_str = self.pdy.get_sizes_str()
        s_str = self.sdy.get_levels_sizes_str(position1, position2)
        res = '\n\n'.join([p_str, s_str])
        return res
    
    def get_bottleneck_position(self):
        return self.sdy.get_bottleneck_position()