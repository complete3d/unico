import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from .build import MODELS
from models.Transformer_utils import *
from utils import misc
import torch.nn.functional as F
import copy
from scipy.optimize import linear_sum_assignment
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL1_, ChamferDistanceL1_instance2

def batch_dice_loss(inputs, targets, unique_gt_indices):
    id2idx = {v.item(): i for i, v in enumerate(unique_gt_indices)}
    remapped = torch.tensor([id2idx[t.item()] for t in targets], device=targets.device)
    targets = F.one_hot(remapped, num_classes=len(unique_gt_indices)).permute(1, 0).float()  # shape: [m, N]
    inputs = inputs.sigmoid() # m, 512
    numerator = 2 * torch.einsum("mc,nc->mn", inputs, targets)  # m, 512
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]  # m, 25
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss 


def batch_sigmoid_ce_loss(inputs, targets, unique_gt_indices):
    point_query_number = inputs.shape[1] 
    id2idx = {v.item(): i for i, v in enumerate(unique_gt_indices)}
    remapped = torch.tensor([id2idx[t.item()] for t in targets], device=targets.device)
    targets = F.one_hot(remapped, num_classes=len(unique_gt_indices)).permute(1, 0).float()  # shape: [m, N]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    return loss / point_query_number


class SelfAttnBlockApi(nn.Module):
    r'''
        1. Norm Encoder Block 
            block_style = 'attn'
        2. Concatenation Fused Encoder Block
            block_style = 'attn-deform'  
            combine_style = 'concat'
        3. Three-layer Fused Encoder Block
            block_style = 'attn-deform'  
            combine_style = 'onebyone'        
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, block_style='attn-deform', combine_style='concat',
            k=10, n_group=2
        ):

        super().__init__()
        self.combine_style = combine_style
        assert combine_style in ['concat', 'onebyone'], f'got unexpect combine_style {combine_style} for local and global attn'
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()        

        # Api desigin
        block_tokens = block_style.split('-')
        assert len(block_tokens) > 0 and len(block_tokens) <= 2, f'invalid block_style {block_style}'
        self.block_length = len(block_tokens)
        self.attn = None
        self.local_attn = None
        for block_token in block_tokens:
            assert block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect block_token {block_token} for Block component'
            if block_token == 'attn':
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif block_token == 'rw_deform':
                self.local_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'deform':
                self.local_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'graph':
                self.local_attn = DynamicGraphAttention(dim, k=k)
            elif block_token == 'deform_graph':
                self.local_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.attn is not None and self.local_attn is not None:
            if combine_style == 'concat':
                self.merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos, idx=None):
        feature_list = []
        if self.block_length == 2:
            if self.combine_style == 'concat':
                norm_x = self.norm1(x)
                if self.attn is not None:
                    global_attn_feat = self.attn(norm_x)
                    feature_list.append(global_attn_feat)
                if self.local_attn is not None:
                    local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.merge_map(f)
                    x = x + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path3(self.ls3(self.local_attn(self.norm3(x), pos, idx=idx)))

        elif self.block_length == 1:
            norm_x = self.norm1(x)
            if self.attn is not None:
                global_attn_feat = self.attn(norm_x)
                feature_list.append(global_attn_feat)
            if self.local_attn is not None:
                local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                x = x + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
   
class CrossAttnBlockApi(nn.Module):
    r'''
        1. Norm Decoder Block 
            self_attn_block_style = 'attn'
            cross_attn_block_style = 'attn'
        2. Concatenation Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'concat'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'
        3. Three-layer Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'onebyone'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'onebyone'    
        4. Design by yourself
            #  only deform the cross attn
            self_attn_block_style = 'attn'  
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
            #  perform graph conv on self attn
            self_attn_block_style = 'attn-graph'  
            self_attn_combine_style = 'concat'    
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
            self_attn_block_style='attn-deform', self_attn_combine_style='concat',
            cross_attn_block_style='attn-deform', cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        super().__init__()        
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()      

        # Api desigin
        # first we deal with self-attn
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.self_attn_combine_style = self_attn_combine_style
        assert self_attn_combine_style in ['concat', 'onebyone'], f'got unexpect self_attn_combine_style {self_attn_combine_style} for local and global attn'
  
        self_attn_block_tokens = self_attn_block_style.split('-')
        assert len(self_attn_block_tokens) > 0 and len(self_attn_block_tokens) <= 2, f'invalid self_attn_block_style {self_attn_block_style}'
        self.self_attn_block_length = len(self_attn_block_tokens)
        self.self_attn = None
        self.local_self_attn = None
        for self_attn_block_token in self_attn_block_tokens:
            assert self_attn_block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect self_attn_block_token {self_attn_block_token} for Block component'
            if self_attn_block_token == 'attn':
                self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif self_attn_block_token == 'rw_deform':
                self.local_self_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'deform':
                self.local_self_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'graph':
                self.local_self_attn = DynamicGraphAttention(dim, k=k)
            elif self_attn_block_token == 'deform_graph':
                self.local_self_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.self_attn is not None and self.local_self_attn is not None:
            if self_attn_combine_style == 'concat':
                self.self_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Then we deal with cross-attn
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()  

        self.cross_attn_combine_style = cross_attn_combine_style
        assert cross_attn_combine_style in ['concat', 'onebyone'], f'got unexpect cross_attn_combine_style {cross_attn_combine_style} for local and global attn'
        
        # API desigin
        cross_attn_block_tokens = cross_attn_block_style.split('-')
        assert len(cross_attn_block_tokens) > 0 and len(cross_attn_block_tokens) <= 2, f'invalid cross_attn_block_style {cross_attn_block_style}'
        self.cross_attn_block_length = len(cross_attn_block_tokens)
        self.cross_attn = None
        self.local_cross_attn = None
        for cross_attn_block_token in cross_attn_block_tokens:
            assert cross_attn_block_token in ['attn', 'deform', 'graph', 'deform_graph'], f'got unexpect cross_attn_block_token {cross_attn_block_token} for Block component'
            if cross_attn_block_token == 'attn':
                self.cross_attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif cross_attn_block_token == 'deform':
                self.local_cross_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif cross_attn_block_token == 'graph':
                self.local_cross_attn = DynamicGraphAttention(dim, k=k)
            elif cross_attn_block_token == 'deform_graph':
                self.local_cross_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.cross_attn is not None and self.local_cross_attn is not None:
            if cross_attn_combine_style == 'concat':
                self.cross_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm_q_2 = norm_layer(dim)
                self.norm_v_2 = norm_layer(dim)
                self.ls5 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path5 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None, primitive_decoder=False, primitive_mask=None):
        
        if not primitive_decoder:
            if denoise_length is None:
                mask = None  
            else:
                query_len = q.size(1)
                mask = torch.zeros(query_len, query_len).to(q.device)
                mask[:-denoise_length, -denoise_length:] = 1.
            cross_mask = None
        else:
            mask = None
            cross_mask = primitive_mask

        if not primitive_decoder:
            feature_list = []
            if self.self_attn_block_length == 2:
                if self.self_attn_combine_style == 'concat':
                    norm_q = self.norm1(q)
                    if self.self_attn is not None:
                        global_attn_feat = self.self_attn(norm_q, mask=mask)
                        feature_list.append(global_attn_feat)
                    if self.local_self_attn is not None:
                        local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                        feature_list.append(local_attn_feat)
                    # combine
                    if len(feature_list) == 2:
                        f = torch.cat(feature_list, dim=-1)
                        f = self.self_attn_merge_map(f)
                        q = q + self.drop_path1(self.ls1(f))
                    else:
                        raise RuntimeError()
                else: # onebyone
                    q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q), mask=mask)))
                    q = q + self.drop_path3(self.ls3(self.local_self_attn(self.norm3(q), q_pos, idx=self_attn_idx, denoise_length=denoise_length)))

            elif self.self_attn_block_length == 1:
                norm_q = self.norm1(q)
                if self.self_attn is not None:
                    global_attn_feat = self.self_attn(norm_q, mask=mask)
                    feature_list.append(global_attn_feat)
                if self.local_self_attn is not None:
                    local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 1:
                    f = feature_list[0]
                    q = q + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()


            feature_list = []
            if self.cross_attn_block_length == 2:
                if self.cross_attn_combine_style == 'concat':
                    norm_q = self.norm_q(q)
                    norm_v = self.norm_v(v)
                    if self.cross_attn is not None:
                        global_attn_feat = self.cross_attn(norm_q, norm_v, mask =cross_mask)
                        feature_list.append(global_attn_feat)
                    if self.local_cross_attn is not None:
                        local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                        feature_list.append(local_attn_feat)
                    # combine
                    if len(feature_list) == 2:
                        f = torch.cat(feature_list, dim=-1)
                        f = self.cross_attn_merge_map(f)
                        q = q + self.drop_path4(self.ls4(f))
                    else:
                        raise RuntimeError()
                else: # onebyone
                    q = q + self.drop_path4(self.ls4(self.cross_attn(self.norm_q(q), self.norm_v(v))))
                    q = q + self.drop_path5(self.ls5(self.local_cross_attn(q=self.norm_q_2(q), v=self.norm_v_2(v), q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)))

            elif self.cross_attn_block_length == 1:
                norm_q = self.norm_q(q)
                norm_v = self.norm_v(v)
                if self.cross_attn is not None:
                    global_attn_feat = self.cross_attn(norm_q, norm_v, mask =cross_mask)
                    feature_list.append(global_attn_feat)
                if self.local_cross_attn is not None:
                    local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 1:
                    f = feature_list[0]
                    q = q + self.drop_path4(self.ls4(f))
                else:
                    raise RuntimeError()

            q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
            return q
        else:
            feature_list = []
            if self.cross_attn_block_length == 2:
                if self.cross_attn_combine_style == 'concat':
                    norm_q = self.norm_q(q)
                    norm_v = self.norm_v(v)
                    if self.cross_attn is not None:
                        global_attn_feat = self.cross_attn(norm_q, norm_v, mask =cross_mask)
                        feature_list.append(global_attn_feat)
                    if self.local_cross_attn is not None:
                        local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                        feature_list.append(local_attn_feat)
                    # combine
                    if len(feature_list) == 2:
                        f = torch.cat(feature_list, dim=-1)
                        f = self.cross_attn_merge_map(f)
                        q = q + self.drop_path4(self.ls4(f))
                    else:
                        raise RuntimeError()
                else: # onebyone
                    q = q + self.drop_path4(self.ls4(self.cross_attn(self.norm_q(q), self.norm_v(v))))
                    q = q + self.drop_path5(self.ls5(self.local_cross_attn(q=self.norm_q_2(q), v=self.norm_v_2(v), q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)))

            elif self.cross_attn_block_length == 1:
                norm_q = self.norm_q(q)
                norm_v = self.norm_v(v)
                if self.cross_attn is not None:
                    global_attn_feat = self.cross_attn(norm_q, norm_v, mask =cross_mask)
                    feature_list.append(global_attn_feat)
                if self.local_cross_attn is not None:
                    local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 1:
                    f = feature_list[0]
                    q = q + self.drop_path4(self.ls4(f))
                else:
                    raise RuntimeError()

            q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
            
            feature_list = []
            if self.self_attn_block_length == 2:
                if self.self_attn_combine_style == 'concat':
                    norm_q = self.norm1(q)
                    if self.self_attn is not None:
                        global_attn_feat = self.self_attn(norm_q, mask=mask)
                        feature_list.append(global_attn_feat)
                    if self.local_self_attn is not None:
                        local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                        feature_list.append(local_attn_feat)
                    # combine
                    if len(feature_list) == 2:
                        f = torch.cat(feature_list, dim=-1)
                        f = self.self_attn_merge_map(f)
                        q = q + self.drop_path1(self.ls1(f))
                    else:
                        raise RuntimeError()
                else: # onebyone
                    q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q), mask=mask)))
                    q = q + self.drop_path3(self.ls3(self.local_self_attn(self.norm3(q), q_pos, idx=self_attn_idx, denoise_length=denoise_length)))

            elif self.self_attn_block_length == 1:
                norm_q = self.norm1(q)
                if self.self_attn is not None:
                    global_attn_feat = self.self_attn(norm_q, mask=mask)
                    feature_list.append(global_attn_feat)
                if self.local_self_attn is not None:
                    local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 1:
                    f = feature_list[0]
                    q = q + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            return q

######################################## Entry ########################################  
class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        block_style_list=['attn-deform'], combine_style='concat', k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SelfAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                block_style=block_style_list[i], combine_style=combine_style, k=k, n_group=n_group
            ))

    def forward(self, x, pos):
        idx = idx = knn_point(self.k, pos, pos)
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx) 
        return x

class TransformerDecoder(nn.Module):
    """ Transformer Decoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
        cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
        k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(CrossAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                self_attn_block_style=self_attn_block_style_list[i], self_attn_combine_style=self_attn_combine_style,
                cross_attn_block_style=cross_attn_block_style_list[i], cross_attn_combine_style=cross_attn_combine_style,
                k=k, n_group=n_group
            ))

    def forward(self, q, v, q_pos, v_pos, denoise_length=None, primitive_decoder=False, primitive_mask=None):
        if not primitive_decoder:
            if denoise_length is None:
                self_attn_idx = knn_point(self.k, q_pos, q_pos)
            else:
                self_attn_idx = None
            cross_attn_idx = knn_point(self.k, v_pos, q_pos)
            for _, block in enumerate(self.blocks):
                q = block(q, v, q_pos, v_pos, self_attn_idx=self_attn_idx, cross_attn_idx=cross_attn_idx, denoise_length=denoise_length)
        else:
            self_attn_idx = None
            cross_attn_idx = None
            for _, block in enumerate(self.blocks):
                q = block(q, v, q_pos, v_pos, self_attn_idx=self_attn_idx, cross_attn_idx=cross_attn_idx, denoise_length=denoise_length, primitive_decoder=primitive_decoder, primitive_mask=primitive_mask)
        return q

class PointTransformerEncoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    Args:
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        init_values: (float): layer-scale init values
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
        act_layer: (nn.Module): MLP activation layer
    """
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            block_style_list=['attn-deform'], combine_style='concat',
            k=10, n_group=2
        ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(block_style_list) == depth
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            block_style_list=block_style_list,
            combine_style=combine_style,
            k=k,
            n_group=n_group)
        self.norm = norm_layer(embed_dim) 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos):
        x = self.blocks(x, pos)
        return x

class PointTransformerDecoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    """
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        """
        Args:
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(self_attn_block_style_list) == len(cross_attn_block_style_list) == depth
        self.blocks = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            self_attn_block_style_list=self_attn_block_style_list, 
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style_list=cross_attn_block_style_list, 
            cross_attn_combine_style=cross_attn_combine_style,
            k=k, 
            n_group=n_group
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None, primitive_decoder=False, primitive_mask=None):
    
        q = self.blocks(q, v, q_pos, v_pos, denoise_length=denoise_length, primitive_decoder=primitive_decoder, primitive_mask = primitive_mask)
        return q

class PointTransformerEncoderEntry(PointTransformerEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

class PointTransformerDecoderEntry(PointTransformerDecoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

######################################## Grouper ########################################  
class DGCNN_Grouper(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        print('using group version 2')
        self.k = k
        # self.knn = KNN(k=k, transpose_mode=False)
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 32),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )
        self.num_features = 128
    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            # _, idx = self.knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, num):
        '''
            INPUT:
                x : bs N 3
                num : list e.g.[1024, 512]
            ----------------------
            OUTPUT:

                coor bs N 3
                f    bs N C(128) 
        '''
        x = x.transpose(-1, -2).contiguous()

        coor = x
        f = self.input_trans(x)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[1])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()

        return coor, f

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class SimpleEncoder(nn.Module):
    def __init__(self, k = 32, embed_dims=128):
        super().__init__()
        self.embedding = Encoder(embed_dims)
        self.group_size = k

        self.num_features = embed_dims

    def forward(self, xyz, n_group):
        # 2048 divide into 128 * 32, overlap is needed
        if isinstance(n_group, list):
            n_group = n_group[-1] 

        center = misc.fps(xyz, n_group) # B G 3
            
        assert center.size(1) == n_group, f'expect center to be B {n_group} 3, but got shape {center.shape}'
        
        batch_size, num_points, _ = xyz.shape
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == n_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, n_group, self.group_size, 3).contiguous()
            
        assert neighborhood.size(1) == n_group
        assert neighborhood.size(2) == self.group_size
            
        features = self.embedding(neighborhood) # B G C
        
        return center, features

######################################## Fold ########################################    
class Fold(nn.Module):
    def __init__(self, in_channel, step , hidden_dim=512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2

class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.layer = Mlp(self.input_dims, hidden_dim, step * 3)

    def forward(self, rec_feature):
        '''
        Input BNC
        '''
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature
            
        patch_feature = torch.cat([
                g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
                token_feature
            ], dim = -1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step , 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc

######################################## PCTransformer ########################################   
class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num  = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        print_log(f'Transformer with config {config}', logger='MODEL')
        # base encoder
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k = 16)
        else:
            self.grouper = SimpleEncoder(k = 32, embed_dims=512)
        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, encoder_config.embed_dim)
        )  
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, encoder_config.embed_dim)
        )
        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry(encoder_config)

        self.increase_dim = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))
        # query generator
        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, decoder_config.embed_dim)
        )
        # assert decoder_config.embed_dim == encoder_config.embed_dim
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)
        # Coarse Level 2 : Decoder
        self.decoder = PointTransformerDecoderEntry(decoder_config)
 
        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz):
        bs = xyz.size(0)
        coor, f = self.grouper(xyz, self.center_num) # b n c
        pe =  self.pos_embed(coor)
        x = self.input_proj(f)

        x = self.encoder(x + pe, coor) # b n c
        global_feature = self.increase_dim(x) # B 1024 N 
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)

        coarse_inp = misc.fps(xyz, self.num_query//2) # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1) # B 224+128 3?

        mem = self.mem_link(x)

        # query selection
        query_ranking = self.query_ranking(coarse) # b n 1
        idx = torch.argsort(query_ranking, dim=1, descending=True) # b n 1
        coarse = torch.gather(coarse, 1, idx[:,:self.num_query].expand(-1, -1, coarse.size(-1)))

        if self.training:
            # add denoise task
            # first pick some point : 64?
            picked_points = misc.fps(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            coarse = torch.cat([coarse, picked_points], dim=1) # B 256+64 3?
            denoise_length = 64     

            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length)

            return q, coarse, denoise_length

        else:
            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c
            
            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)

            return q, coarse, 0



############################# Multi-primitive segmentation #############################
class Primitive_Segmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.primitive_query_type = config.primitive_query_type
        assert self.primitive_query_type in ['static', 'dynamic'], f'unexpected primitive_query_type {self.primitive_query_type}'
        self.primitive_query_num = config.primitive_query_num
        primitive_decoder_cofig = config.primitive_decoder_cofig
        hidden_dim = primitive_decoder_cofig.embed_dim
        self.num_decoders = primitive_decoder_cofig.depth
        self.second_stage = getattr(config, 'second_stage', 300)

        # Homogeneous Cartesian quadric with 10 parameters: Ax² + By² + Cz² + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz + J = 0
        # or plane with 4 parameters: 2Gx + 2Hy + 2Iz + J = 0 (special case of quadric)
        self.num_quadric_params = getattr(config, 'num_quadric_params', 10)
     
        if self.primitive_query_type == 'static':
            self.query_feat = nn.Embedding(self.primitive_query_num, hidden_dim) 
        elif self.primitive_query_type == 'dynamic':
            self.pos_embed = nn.Sequential(
                nn.Linear(3, 128),  
                nn.GELU(),
                nn.Linear(128, hidden_dim)
            )
            self.query_feat = nn.Embedding(self.primitive_query_num, hidden_dim)
        
        # Multi-class head: 4 primitive types + background (index 4) or plane only + background
        self.num_primitive_classes = getattr(config, 'num_primitive_classes', 5)
        self.class_embed_head = nn.Linear(hidden_dim, self.num_primitive_classes)
        self.mask_embed_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.GELU(),
                                             nn.Linear(hidden_dim, hidden_dim)) 
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.quadric_head = nn.Linear(hidden_dim, self.num_quadric_params)
        
        self.point_feature_encoder =  nn.Sequential(nn.Linear(384, hidden_dim),
                                                    nn.GELU(),
                                                    nn.Linear(hidden_dim, hidden_dim)) 
        
        self.decoder_layers = nn.ModuleList()
        for i in range(self.num_decoders):
            layer_config = copy.deepcopy(primitive_decoder_cofig)
            layer_config.depth = 1
            layer_config.self_attn_block_style_list = [primitive_decoder_cofig.self_attn_block_style_list[i]]
            layer_config.cross_attn_block_style_list = [primitive_decoder_cofig.cross_attn_block_style_list[i]]
            self.decoder_layers.append(PointTransformerDecoderEntry(layer_config))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def mask_module(self, query_feat, mask_features):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        output_class = self.class_embed_head(query_feat)
        output_masks = torch.einsum('bic,bjc->bij', mask_embed, mask_features)
        
        return output_class, output_masks

    @staticmethod
    def _canonicalize_quadrics(params: torch.Tensor) -> torch.Tensor:
        """Normalize quadric coefficient vectors and fix sign ambiguity.
        Args:
            params: (B, Q, 10)
        Returns:
            canonical (B, Q, 10) with ||v||=1 and last coefficient <= 0.
        """
        # Sign-sensitive mode: only scale-normalize; retain raw orientation
        norm = params.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return params / norm


    def get_primitive_query(self, ret):
        """
        Fit homogeneous Cartesian quadric surfaces to segmented point clouds.
        Used ONLY as a metric (not for training gradients). Stores result in
        ret['quadrics_fitted'] without overwriting the regressed 'quadrics'.
        """
        pred_masks = ret['pred_masks'].sigmoid()  
        rebuild_points = ret['rebuild_points']    
        factor = rebuild_points.shape[1] // pred_masks.shape[2]  
        pred_masks = pred_masks.unsqueeze(-1).expand(-1, -1, -1, factor).reshape(
            pred_masks.shape[0], pred_masks.shape[1], -1)  
        return ret
    
    def forward(self, ret, epoch=1):
        coarse_point_cloud, rebuild_points, point_features = ret["coarse_point_cloud"], ret["rebuild_points"], ret["point_features"]
        point_features = self.point_feature_encoder(point_features)
        batch_size = coarse_point_cloud.shape[0]

        if self.primitive_query_type == 'static':
            queries = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            pos = misc.fps(coarse_point_cloud, self.primitive_query_num)
            pos_embed = self.pos_embed(pos)
            query = self.query_feat.weight.unsqueeze(0)
            queries = query + pos_embed

        for decoder_counter in range(self.num_decoders):
            queries = self.decoder_layers[decoder_counter](queries, point_features, q_pos=None, v_pos=None, primitive_decoder=True, primitive_mask=None)

        output_class, output_masks = self.mask_module(queries, point_features)

        # Direct regression of quadric coefficients (always available for losses / downstream)
        predicted_quadrics = self._canonicalize_quadrics(self.quadric_head(queries))  # B, num_queries, 10

        if self.training:
            ret = {
                "coarse_point_cloud": coarse_point_cloud,  # B, 512, 3
                "rebuild_points": rebuild_points,          # B, 8192, 3
                "class_prob": output_class,                # B, 40, 5 (plane,cylinder,sphere,cone,background=4)
                "pred_masks": output_masks,                # B, 40, 512
                "denoised_coarse": ret["denoised_coarse"], # B, 64, 3
                "denoised_fine": ret["denoised_fine"],     # B, 1024, 3
                "quadrics": predicted_quadrics             # regressed quadric coefficients (training signal)
            }
        else:
            ret = {
                "coarse_point_cloud": coarse_point_cloud,
                "rebuild_points": rebuild_points,
                "class_prob": output_class,
                "pred_masks": output_masks,
                "quadrics": predicted_quadrics
            }
        
        if epoch >= self.second_stage:
            ret = self.get_primitive_query(ret)

        return ret


@MODELS.register_module()
class UNICO(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.num_points = getattr(config, 'num_points', None)
        self.primitive_query_num = getattr(config, 'primitive_query_num', 40)
        
        self.decoder_type = config.decoder_type
        self.first_stage = getattr(config, 'first_stage', 200)
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'

        self.fold_step = 8
        self.base_model = PCTransformer(config)
        
        if self.decoder_type == 'fold':
            self.factor = self.fold_step**2
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)  # rebuild a cluster point
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_query
                assert self.num_points % self.num_query == 0
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.num_points // self.num_query)  # rebuild a cluster point
            else:
                self.factor = self.fold_step**2
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step**2)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        
        # Choose between original plane segmentation or multi-primitive segmentation
        if hasattr(config, 'primitive_segmentation_config'):
            self.primitive_segmentation = Primitive_Segmentation(config.primitive_segmentation_config)
            self.use_multi_primitive = getattr(config, 'use_multi_primitive', True)
        
        
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()
        self.get_index = ChamferDistanceL1_()
        self.chamfer_primitive  = ChamferDistanceL1_instance2()

        
    def get_segmentation_labels(self, ret, gt, gt_index, coarse=False):
        """
        Generate segmentation labels based on predicted masks.
        coarse: If True, use coarse point cloud; otherwise, use rebuilt points. 
                try not to use coarse point cloud, it is not accurate enough.

        Args:
            ret: Dictionary containing 'pred_masks' with shape (B, N1, N2)

        Returns:
            seg_labels: Tensor of shape (B, N1) with segmentation labels
        """
        if coarse:
            coarse_point_cloud = ret['coarse_point_cloud']  # B, N1, 3
            idx = self.get_index(coarse_point_cloud, gt).long() # B, N1
            B, N = idx.shape  # B=5, N=512
            batch_indices = torch.arange(B).unsqueeze(1).expand(-1, N)  # shape [5, 512]
            selected_gt = gt_index[batch_indices, idx]  # shape [5, 512]
            ret["instance_lables"] = selected_gt
        else:
            points = ret['rebuild_points']  # B, N1, 3
            idx = self.get_index(points, gt).long()  # B, N1
            B, N = idx.shape  # B=5, N=16384
            batch_indices = torch.arange(B).unsqueeze(1).expand(-1, N)
            selected_gt = gt_index[batch_indices, idx]  # shape [5, 16384]
            selected_gt = selected_gt.reshape(B, -1, self.factor)
            patch_index = torch.mode(selected_gt, dim=-1).values
            ret["instance_lables"] = patch_index

        return ret
    
    def _quadric_param_cost_matrix(
        self,
        theta_pred: torch.Tensor,   # [Np,10] -> [A,B,C,D,E,F,G,H,I,J]
        theta_gt:   torch.Tensor,   # [Ng,10]
        eps: float = 1e-9,
        beta: float = 1e-2,         # Smooth-L1 (Huber) delta
    ):
        """
        Uniform, scale-invariant, sign-sensitive quadric supervision.

        Canonicalization (both pred & GT):  θ̂ = θ / (||θ||₂ + eps)   (no sign flip)
        Cost: elementwise Smooth-L1(θ̂_pred - θ̂_gt), averaged over the 10 terms.
        Returns: [Np, Ng]
        """
        import torch
        import torch.nn.functional as F


        # ---- canonicalization (scale invariance, sign supervised)----
        s_p = torch.linalg.vector_norm(theta_pred, dim=1).clamp_min(eps)   # [Np]
        s_g = torch.linalg.vector_norm(theta_gt,   dim=1).clamp_min(eps)   # [Ng]
        vec_p = theta_pred / s_p[:, None]   # [Np,10/4]
        vec_g = theta_gt   / s_g[:, None]   # [Ng,10/4]

        # ---- pairwise Smooth-L1 on vectors ----
        # (θ̂_p - θ̂_g) -> Huber -> mean over terms
        d = vec_p[:, None, :] - vec_g[None, :, :]                 # [Np,Ng,10/4]
        costs = F.smooth_l1_loss(d, torch.zeros_like(d), beta=beta, reduction='none').mean(dim=2)  # [Np,Ng]

        if self.training:
            if not torch.isfinite(costs).all():
                bad = torch.nonzero(~torch.isfinite(costs), as_tuple=False)[:5]
                raise ValueError(f"Non-finite costs detected (first 5): {bad.tolist()}")

        return costs


    def get_loss(self, config, ret, gt, gt_index, gt_coeff, gt_type, epoch = 1):
        """
        Compute losses for the model

        Args:
            config: Configuration with loss weights
            ret: prediction
            gt: GT points
            gt_index: GT indices
            gt_coeff: GT coefficients
            gt_type: GT types

        Returns:
            Dictionary of computed losses
        """
        # Deterministic: dataset supplies gt_type with a trailing singleton dim -> remove it once.
        gt_type = gt_type.squeeze(-1)
        reconstructed_points, coarse_point_cloud = ret['rebuild_points'],  ret['coarse_point_cloud']
        
        device = reconstructed_points.device
        losses = {}
        if epoch < config['first_stage'] or epoch >= config['second_stage']:
            
            if self.training:
                denoised_coarse, denoised_fine = ret['denoised_coarse'], ret['denoised_fine']
                idx = knn_point(self.factor, gt, denoised_coarse) # B n k 
                denoised_target = index_points(gt, idx) # B n k 3 
                denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
                assert denoised_target.size(1) == denoised_fine.size(1)
                loss_denoised = self.loss_func(denoised_fine, denoised_target)
                loss_denoised = loss_denoised * 0.5
                losses["loss_denoised"] = loss_denoised

            loss_coarse = self.loss_func(coarse_point_cloud, gt)
            loss_fine = self.loss_func(reconstructed_points, gt)
            loss_recon = loss_coarse + loss_fine
            losses["loss_coarse"] = loss_coarse
            losses["chamfer_norm1_loss"] = loss_fine
            if epoch < config['first_stage'] and self.training:
                losses["total_loss_stage1"] = loss_recon + loss_denoised
            
        if epoch >= config['second_stage'] and epoch < config['third_stage']:
            ret = self.get_segmentation_labels(ret, gt, gt_index, coarse=False)  # patch-level labels only
            losses["primitive_chamfer_loss"] = losses.get("primitive_chamfer_loss", 0.0)
            losses["classification_loss"] = losses.get("classification_loss", 0.0)
            losses["primitive_normal_loss"] = losses.get("primitive_normal_loss", 0.0)
            losses["mask_loss"] = losses.get("mask_loss", 0.0)
            losses["dice_loss"] = losses.get("dice_loss", 0.0)
            
            class_prob, reconstructed_points, pred_masks, predicted_quadrics  = ret['class_prob'], ret['rebuild_points'], ret['pred_masks'], ret['quadrics']
            tgt_ids = ret["instance_lables"]
            batch_size = class_prob.size(0)
            size_total = 0
            for batch_idx in range(batch_size):
                # continue
                unique_gt_indices = torch.unique(gt_index[batch_idx].int())
                unique_gt_indices = unique_gt_indices[unique_gt_indices != -1].long()  
                num_ground_truth_primitives = unique_gt_indices.size(0)

                ground_truth_pointclouds = [gt[batch_idx, (gt_index[batch_idx] == idx)].reshape(-1, 3) for idx in unique_gt_indices]  
                reconstructed_pointclouds = reconstructed_points[batch_idx]
                
                # loss term1: Compute Primitive Chamfer Distance soft labeled
                mask_weights = pred_masks[batch_idx].sigmoid()  # num_queries, 512
                mask_weights = mask_weights.view(self.primitive_query_num, -1, 1).expand(-1, -1, self.factor).reshape(self.primitive_query_num, -1)  # num_queries, 512 * factor
                primitive_chamfer_distance = self.chamfer_primitive(reconstructed_pointclouds, ground_truth_pointclouds, mask_weights)
                
                # loss term2: Primitive parameter cost (type-aware plane vs curved only for now)
                if self.use_multi_primitive:
                    _pred_coeff = predicted_quadrics[batch_idx][..., :10].float()  # [num_queries, 10]
                    _gt_coeff = gt_coeff[batch_idx, unique_gt_indices][..., :10].float()  # [num_gt, 10]
                    _gt_types = gt_type[batch_idx, unique_gt_indices].long()  # 0 plane, 1 cylinder, 2 sphere, 3 cone
                else:
                    _pred_coeff = predicted_quadrics[batch_idx][..., :4].float()  # [num_queries, 4]
                    _gt_coeff = gt_coeff[batch_idx, unique_gt_indices][..., :4].float()  # [num_gt, 4]
                    _gt_types = torch.zeros_like(unique_gt_indices).long()  # treat all GT as planes for now
                primitive_normal_loss = self._quadric_param_cost_matrix(_pred_coeff, _gt_coeff)

                # loss term3 & 4: Multi-class classification (types + background) + mask/dice
                classification_scores = class_prob[batch_idx]  # [Q,5]
                # Use patch-level ids directly for mask/dice (legacy behavior)
                tgt_id = tgt_ids[batch_idx]

                mask_loss = batch_sigmoid_ce_loss(pred_masks[batch_idx], tgt_id, unique_gt_indices)
                dice_loss = batch_dice_loss(pred_masks[batch_idx], tgt_id, unique_gt_indices)

                # Classification cost (multi-class) reintegrated similar to earlier binary object_prob term
                log_probs = classification_scores.log_softmax(dim=-1)  # [Q,5]
                class_cost = -log_probs[:, _gt_types]

                cost_matrix = (class_cost * config.obj_class_loss_weight +
                               primitive_chamfer_distance * config.primitive_chamfer_loss_weight +
                               primitive_normal_loss * config.primitive_normal_loss_weight +
                               mask_loss * config.mask_loss_weight +
                               dice_loss * config.mask_loss_weight)

                
                if not torch.isfinite(cost_matrix).all():
                    bad = ~torch.isfinite(cost_matrix)
                    bad_idx = bad.nonzero()
                    for i in range(min(5, bad_idx.size(0))):
                        pi, gi = bad_idx[i].tolist()
                        print(f"    -> cost_matrix[{pi},{gi}] = {cost_matrix[pi,gi].item()}")
                    cost_matrix = torch.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=1e6)

                hungarian_assignment = linear_sum_assignment(cost_matrix.detach().cpu().numpy()) if num_ground_truth_primitives > 0 else ([], [])
                if num_ground_truth_primitives > 0:
                    hungarian_assignment = [torch.tensor(a, dtype=torch.long, device=device) for a in hungarian_assignment]
                    pred_matched_idx, gt_matched_idx = hungarian_assignment
                    matched_primitive_chamfer_distance = primitive_chamfer_distance[pred_matched_idx, gt_matched_idx]
                    matched_primitive_normal_loss = primitive_normal_loss[pred_matched_idx, gt_matched_idx]
                    mached_mask_loss = mask_loss[pred_matched_idx, gt_matched_idx]
                    matched_dice_loss = dice_loss[pred_matched_idx, gt_matched_idx]
                else:
                    pred_matched_idx = torch.empty(0, dtype=torch.long, device=device)
                    gt_matched_idx = torch.empty(0, dtype=torch.long, device=device)
                    matched_primitive_chamfer_distance = torch.zeros(0, device=device)
                    matched_primitive_normal_loss = torch.zeros(0, device=device)
                    mached_mask_loss = torch.zeros(0, device=device)
                    matched_dice_loss = torch.zeros(0, device=device)
                
                if self.use_multi_primitive:
                    targets = torch.full((self.primitive_query_num,), 4, dtype=torch.long, device=device)
                else:
                    targets = torch.full((self.primitive_query_num,), 1, dtype=torch.long, device=device)
                if pred_matched_idx.numel() > 0:
                    targets[pred_matched_idx] = _gt_types[gt_matched_idx]
                ce_per_query = F.cross_entropy(classification_scores, targets, reduction='none')
                if self.use_multi_primitive:
                    matched_mask_bool = targets != 4
                else:
                    matched_mask_bool = targets != 1
                matched_class_loss = ce_per_query[matched_mask_bool]
                unmatched_class_loss = ce_per_query[~matched_mask_bool] * config.non_obj_class_loss_weight
                total_classification_loss = torch.cat([matched_class_loss, unmatched_class_loss])

                losses["primitive_chamfer_loss"] += matched_primitive_chamfer_distance.sum()
                losses["classification_loss"] += total_classification_loss.sum()
                losses["primitive_normal_loss"] += matched_primitive_normal_loss.sum()
                losses["mask_loss"] += mached_mask_loss.sum()
                losses["dice_loss"] += matched_dice_loss.sum()
                size_total += num_ground_truth_primitives
            
            losses["classification_loss"] /= (batch_size * self.primitive_query_num)
            losses["primitive_chamfer_loss"] /= size_total
            losses["primitive_normal_loss"] /= size_total
            losses["mask_loss"] /= size_total
            losses["dice_loss"] /= size_total
            
            if self.training:
                losses["total_loss_stage3"] = config.obj_class_loss_weight * losses["classification_loss"]  + config.primitive_chamfer_loss_weight * losses["primitive_chamfer_loss"] + config.chamfer_norm1_loss_weight * (losses["chamfer_norm1_loss"] + losses["loss_coarse"] + losses["loss_denoised"]) + config.primitive_normal_loss_weight * losses["primitive_normal_loss"] + config.mask_loss_weight * (losses["mask_loss"] + losses["dice_loss"])
            else:
                losses["total_loss_stage3"] = config.obj_class_loss_weight * losses["classification_loss"]  + config.primitive_chamfer_loss_weight * losses["primitive_chamfer_loss"] + config.primitive_normal_loss_weight * losses["primitive_normal_loss"] + config.chamfer_norm1_loss_weight * (losses["chamfer_norm1_loss"] + losses["loss_coarse"]) + config.mask_loss_weight * (losses["mask_loss"] + losses["dice_loss"])
        return losses
    
    
    def forward(self, xyz, epoch = None):
        q, coarse_point_cloud, denoise_length = self.base_model(xyz) # B M C and B M 3
    
        B, M ,C = q.shape

        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B*M, -1)) # BM C
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)    # B M 3 S
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3)  # B M S 3

        else:
            rebuild_feature = self.reduce_map(rebuild_feature) # B M C
            relative_xyz = self.decode_head(rebuild_feature)   # B M S 3
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3
        
        if self.training:
           
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            ret =  {"coarse_point_cloud": pred_coarse,
                    "rebuild_points": pred_fine,
                    "denoised_coarse": denoised_coarse,
                    "denoised_fine": denoised_fine}
        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query

            ret =  {"coarse_point_cloud": coarse_point_cloud,
                    "rebuild_points": rebuild_points}
        
        if epoch is None or epoch >= self.first_stage:
            ret["point_features"] = q[:, :-denoise_length, :] if self.training else q
            ret = self.primitive_segmentation(ret, epoch=epoch)
        return ret
