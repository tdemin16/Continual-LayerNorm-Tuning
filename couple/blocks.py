from collections import OrderedDict

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    

class WeightSelectionMechanism(nn.Module):
    def __init__(self, args, embed_dim: int) -> None:
        super().__init__()
        self.curr_task = 0
        self.new_key = args.new_key
        self.detach_selection = args.detach_selection
        
        self.custom_match = False
        self.mode = None
        if args.full_match:
            self.mode = 'full_match'

        self.key = nn.Parameter(torch.randn((args.num_tasks, embed_dim)))
        if args.use_att:
            self.att = nn.Parameter(torch.randn((args.num_tasks, embed_dim)))
        if args.key_init == 'uniform':
            nn.init.uniform_(self.key, -1, 1)
            if hasattr(self, 'att'):
                nn.init.uniform_(self.att, -1, 1)
        elif args.key_init == 'orthogonal':
            nn.init.orthogonal_(self.key)
            if hasattr(self, 'att'):
                nn.init.orthogonal_(self.att)
        else:
            raise NotImplementedError(f'{args.key_init} is not a valid key initialization')
    
    @torch.no_grad()
    def get_last_similarity(self) -> int:
        """
        Returns the similarity between the last task key and all the previous ones.
        """
        assert self.curr_task > 0
        c = self.curr_task
        prev_keys = F.normalize(self.key[0:c], dim=-1)
        curr_key = F.normalize(self.key[c], dim=-1)

        sim = curr_key @ prev_keys.T
        return sim

    def forward(self, feats: torch.Tensor, task: int=-1, task_eval: int=None):
        """
        Selects the most similar set of LN weights for each element in the batch.
        """
        metrics = {}
        c = self.curr_task
        if self.detach_selection:
            feats = feats.detach()

        B = feats.size(0)
        # case inference
        if task == -1:
            # [B, 1, D]
            feats = feats.unsqueeze(1) 
            if hasattr(self, 'att'):
                feats = feats * self.att[:c+1].unsqueeze(0)
            # [B, 1, D]
            feats_norm = F.normalize(feats, dim=-1)
            # [1, C+1, D]
            key_norm = F.normalize(self.key[:c+1], dim=-1).unsqueeze(0)
            # [B, C+1]
            sim = torch.sum(feats_norm * key_norm, dim=-1)
            
            # [B]
            if not self.custom_match:
                sim, tasks = torch.max(sim, dim=-1)
            elif self.mode == 'full_match':
                tasks = torch.full((B, ), task_eval)
                sim = sim[task_eval]
            else:
                raise NotImplementedError(f'custom_match {self.custom_match} - mode {self.mode}')

            metrics['sim'] = sim
            metrics['acc_sim'] = (tasks == task_eval).float().mean()
            metrics['pred'] = tasks.detach().clone().cpu()

        # case training
        else:
            if hasattr(self, 'att'):
                feats = feats * self.att[c].unsqueeze(0)
            feats_norm = F.normalize(feats, dim=-1)
            key_norm = F.normalize(self.key[c], dim=-1).unsqueeze(0)
            sim = torch.sum(feats_norm * key_norm) / B
            tasks = torch.full((B, ), task)

            metrics['sim'] = sim
            # trick to avoid changing the code to much
            metrics['acc_sim'] = torch.tensor([0])
        
        return tasks, metrics
    
    def next_task(self, *args, **kwargs):
        self.curr_task += 1
        c = self.curr_task

        if not self.new_key:
            with torch.no_grad():
                if self.key.grad is not None:
                    self.key.grad.zero_()
                self.key[c] = self.key[c-1]
                if hasattr(self, 'att'):
                    if self.att.grad is not None:
                        self.att.grad.zero_()
                    self.att[c] = self.att[c-1]

    def set_custom_match(self, mode: bool):
        self.custom_match = mode


class MultiTaskLN(nn.Module):
    def __init__(self, args, norm_layer: nn.LayerNorm,
                 w: nn.Parameter, b: nn.Parameter) -> None:
        super().__init__()
        self.curr_task = 0
        
        embed_dim = w.size(0)
        self.ln = norm_layer(embed_dim, elementwise_affine=False)

        self.w = nn.Parameter(w.unsqueeze(0).expand(args.num_tasks, -1))
        self.b = nn.Parameter(b.unsqueeze(0).expand(args.num_tasks, -1))

        self.weight_selection = WeightSelectionMechanism(args, embed_dim)

    def forward(self, x, tasks: torch.Tensor, task_eval: int=-1):
        assert x.size(0) == tasks.size(0)
        metrics = {}
        if hasattr(self, 'weight_selection'):
            feats = x[:, 0]
            tasks, metrics = self.weight_selection(feats, tasks[0].item(), task_eval)
        
        B, P, D = x.size()
        w = self.w[tasks].unsqueeze(1).expand(-1, P, -1)
        b = self.b[tasks].unsqueeze(1).expand(-1, P, -1)
        return self.ln(x) * w + b, metrics
    
    def next_task(self, *args, **kwargs):
        self.curr_task += 1
        c = self.curr_task
        if hasattr(self, 'weight_selection'):
            self.weight_selection.next_task()

        with torch.no_grad():
            if self.w.grad is not None:
                self.w.grad.zero_()
            if self.b.grad is not None:
                self.b.grad.zero_()

            self.w[c] = self.w[c-1]
            self.b[c] = self.b[c-1]

    def set_custom_match(self, mode: bool):
        if hasattr(self, 'wsm'):
            self.wsm.set_custom_match(mode)


class BlockLN(nn.Module):
    """
    Custom Block. It is designed to load original weights but then using custom ones in 
    layer norm scale and shift operations.
    """
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, tasks: torch.Tensor, task_eval: int=None):
        x_norm, m1 = self.mln1(x, tasks, task_eval)
        x = x + self.drop_path1(self.ls1(self.attn(x_norm)))
        x_norm, m2 = self.mln2(x, tasks, task_eval)
        x = x + self.drop_path2(self.ls2(self.mlp(x_norm)))
        
        if m1 == {}:
            return x
        else:
            metrics = {k: (m1[k] + m2[k]) / 2 for k in m1.keys()}
            return x, metrics
    
    def init_params(self, args, norm_layer: nn.LayerNorm):
        self.mln1 = MultiTaskLN(args, norm_layer=norm_layer, w=self.norm1.weight, b=self.norm1.bias)
        self.mln2 = MultiTaskLN(args, norm_layer=norm_layer, w=self.norm2.weight, b=self.norm2.bias)
        
    def next_task(self, *args, **kwargs):
        self.mln1.next_task()
        self.mln2.next_task()

    def set_custom_match(self, mode: bool):
        self.mln1.set_custom_match(mode)
        self.mln2.set_custom_match(mode)
    

class ResPostBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.init_values = init_values

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x):
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class ParallelBlock(nn.Module):

    def __init__(
            self, dim, num_heads, num_parallel=2, mlp_ratio=4., qkv_bias=False, init_values=None,
            drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('attn', Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))
            self.ffns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('mlp', Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))

    def _forward_jit(self, x):
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x):
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x):
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)
        