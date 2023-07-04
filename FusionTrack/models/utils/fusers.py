import torch
import torch.nn as nn

from einops import rearrange, reduce, repeat

from mmcv.runner import BaseModule
from mmcv.cnn import build_activation_layer, build_norm_layer

import copy

from .embeds import ScoreEmbed

class MHSAWarper(nn.MultiheadAttention):
    """A warpper for Multi-Head Self Attention"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, query):
        res, attn = super().forward(query, query, query)
        if getattr(self, "debug", False):
            self.last_attn_map = attn
        return res


class MultiLevelQueryFuser(BaseModule):
    """fuse #Layer x Q x D queries to appearance feat Q x D"""
    def __init__(
        self, 
        d_query,
        d_out,
        d_ffn,
        num_layers=2,
        n_heads=8,
        n_intermidiates=7,  # input + intermidiates
        ffn_act=dict(type="GELU"),
        ffn_norm=dict(type="LN"),
        dropout=0.1,
        use_score_embed=True,
        only_last_query=False
    ) -> None:
        super().__init__()
        self.use_score_embed = use_score_embed
        self.only_last_query = only_last_query
        self.num_layers = num_layers
        if only_last_query:
            n_intermidiates = 2
        if use_score_embed:
            self.score_embed = ScoreEmbed(d_query)
            self.pos_embed = nn.Parameter(torch.randn(n_intermidiates + 1, d_query))
        def build_ffn():
            return nn.Sequential(
                nn.Linear(d_query, d_ffn),
                build_activation_layer(ffn_act),
                nn.Linear(d_ffn, d_query),
            )
        def build_post_ffn():
            return nn.Sequential(
                build_norm_layer(ffn_norm, d_query)[1],
                nn.Dropout(dropout)
            )
        mix_modules = []
        mix_norm_modules = []
        ffn_modules = []
        ffn_norm_modules = []
        self.num_layers = num_layers
        for _ in range(num_layers):
            mix_modules.append(MHSAWarper(d_query, n_heads, dropout=dropout))
            mix_norm_modules.append(build_norm_layer(ffn_norm, d_query)[1])
            ffn_modules.append(build_ffn())
            ffn_norm_modules.append(build_post_ffn())
        self.mix_modules = nn.ModuleList(mix_modules)
        self.mix_norm_modules = nn.ModuleList(mix_norm_modules)
        self.ffn_modules = nn.ModuleList(ffn_modules)
        self.ffn_norm_modules = nn.ModuleList(ffn_norm_modules)
        self.proj_out = nn.Sequential(
            nn.Linear(d_query, d_out),
            build_norm_layer(ffn_norm, d_out)[1],
            nn.Dropout(dropout),
        )
        
    def forward(self, queries, scores):
        """fuse queries to get appearance feat

        Args:
            queries (torch.Tensor): L x B x Q x D
            scores (torch.Tensor): B x Q x 1
        """
        L, B, Q, D = queries.shape
        queries = rearrange(queries, "l b q d -> l (b q) d")
        if self.only_last_query:
            queries = queries[[0, -1]]
        if self.use_score_embed:
            scores = rearrange(scores, "b q c -> 1 (b q) c")
            feats = repeat(self.pos_embed, "n d -> n bq d", bq=B*Q)
            feats = torch.cat(
                [feats[:-1] + queries, feats[-1] + self.score_embed(scores)],
                dim=0
            )
        else:
            feats = queries
        
        for layer in range(self.num_layers):
            feats = feats + self.mix_modules[layer](feats)
            feats = self.mix_norm_modules[layer](feats)
            feats = feats + self.ffn_modules[layer](feats)
            feats = self.ffn_norm_modules[layer](feats)
        
        feats = self.proj_out(feats[-1])
        return rearrange(feats, "(b q) d -> b q d", b=B, q=Q)


FUSERS = {
    "MultiLevelQueryFuser": MultiLevelQueryFuser,
}

def build_fuser(cfg):
    cfg = copy.deepcopy(cfg)
    ftype = cfg.pop("type")
    return FUSERS[ftype](**cfg)