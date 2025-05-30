import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
from functools import partial
import numpy as np
from typing import Optional, Tuple
#from utils.transforms import ResizeLongestSide
from PIL import Image




def interpolate_tensor(input_tensor, size):
    """
    使用线性插值来改变输入 tensor 的大小。
    
    Args:
        input_tensor (torch.Tensor): 输入的二维 tensor，形状为 (hidden_size, hidden_dim)
        size (tuple): 输出 tensor 的大小，形状为 (hidden_size_, hidden_dim)
        
    Returns:
        torch.Tensor: 插值后的输出 tensor
    """
    # 添加两个维度来适应 interpolate 函数
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    
    # 计算插值尺寸
    target_size = [input_tensor.size(2) * size[0] // input_tensor.size(2), 
                   input_tensor.size(3) * size[1] // input_tensor.size(3)]
    
    # 使用线性插值
    output_tensor = F.interpolate(input_tensor, size=target_size, mode='bilinear', align_corners=False)
    
    # 移除添加的维度
    output_tensor = output_tensor.squeeze(0).squeeze(0)
    
    return output_tensor

class AdapterDecoder_Layer(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=0.25, norm_layer = nn.LayerNorm, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        #hidden_dim = int(embed_dim * mlp_ratio)
        hidden_dim = int(embed_dim * embed_dim)
        self.embed_dim = embed_dim
        
        self.norm = norm_layer(embed_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim, bias=False),
                nn.Sigmoid()
        )

        self.spatial = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.ReLU(),
        )
        
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
       
    def forward(self, x):

        hidden_dim,hidden_size = x.size(0),x.size(1)
        x = interpolate_tensor(x,(hidden_dim,int(hidden_size**0.5)**2))
       
        x = x.unsqueeze(0).view(1,hidden_dim,int(hidden_size**0.5),int(hidden_size**0.5))
        
  
        B, C, _, _ = x.size()

        x_channel = self.channel(self.avg_pool(x).view(B, C)).view(B, C, 1, 1) * x
        x_spatial = self.spatial(x_channel)
        
        if self.skip_connect:
            if x_spatial.shape!=x.shape:
                x_spatial = F.interpolate(x_spatial,x.shape[-2:],mode='bilinear',align_corners=False)
            x = x + x_spatial
        else:
            if x_spatial.shape!=x.shape:
                x_spatial = F.interpolate(x_spatial,x.shape[-2:],mode='bilinear',align_corners=False)
            x = x_spatial
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = x.permute(0,3,1,2)
        _,_,h,w = x.size()
        x = x.squeeze(0).view(hidden_dim,h*w)
        x = interpolate_tensor(x,(hidden_dim,hidden_size))
        return x

        
class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, num_local_experts, num_experts_per_tok, router_jitter_noise ,hidden_dim):
        super().__init__()
        
        self.num_experts = num_local_experts
        self.top_k = num_experts_per_tok
        self.hidden_dim = hidden_dim        
        #self.expert = UNet()
        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        #self.unet = UNet(hidden_dim,hidden_dim)
        self.experts = nn.ModuleList([AdapterDecoder_Layer(32)  for _ in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = router_jitter_noise
        self.out_gate = nn.Linear(self.num_experts,self.hidden_dim,bias=False)
        


    def forward(self, hidden_states: torch.Tensor,training=True) -> torch.Tensor:
        """ """

        b,c,h,w = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1)  #B,H,W,C

        hidden_states = hidden_states.view(hidden_states.size(0), -1, hidden_states.size(3))  # B, H*W, C
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if training and self.jitter_noise > 0:
            hidden_states =hidden_states*torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
  
        #hidden_states = hidden_states.view(-1, hidden_dim)
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        #router_logits_ = hidden_states

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        #routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        #pdb.set_trace()
        for expert_idx in range(self.num_experts):
            
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
        
            if len(top_x) == 0:
                continue
            
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
   
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            
            current_state = current_state.permute(1,0)
      
            current_hidden_states = expert_layer(current_state).permute(1,0) * routing_weights[top_x, idx, None]
            
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
   
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.reshape(b,c,h,w)
        
        return final_hidden_states

if __name__ == '__main__':
    x = torch.randn(1,32,64,64)
    Moe = MixtralSparseMoeBlock(4, 2, 0.01,32)
    x_o = Moe(x)
    print(x_o.shape)
