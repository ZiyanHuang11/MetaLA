import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from megatron import mpu
from megatron.model.activations import get_activation

from einops import rearrange
import triton 
import triton.language as tl

from fla.ops.gla import fused_chunk_gla, chunk_gla, fused_recurrent_gla
from megatron.model.norms import LayerNorm, get_norm
from causal_conv1d import causal_conv1d_fn
import einops

class LLaMAParallelMLP(nn.Module):
    """LLaMA's MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Note: multiple_of is used to compute the hidden dimension of the MLP
    """

    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        parallel_output=False,
        multiple_of=256,
        MOE=False,
        MoE_mp_size=1,
    ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation

        self.multiple_of = multiple_of

        # Allow custom intermediate size, e.g. for Mistral
        if neox_args.intermediate_size is not None:
            ff_dim = neox_args.intermediate_size
        else:
            ff_dim = int(2 * neox_args.hidden_size * 4 / 3)
            ff_dim = self.multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.w1 = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )
        self.w3 = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )
        self.w2 = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=False,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )

    def forward(self, hidden_states):
        w1_out, _ = self.w1(hidden_states)
        w3_out, _ = self.w3(hidden_states)
        return self.w2(self.activation_func(w1_out) * w3_out)
      
    
class ParallelMetaLA_Attention_selfaug(nn.Module):
    def __init__(self, neox_args, init_method, output_layer_init_method,):
        super().__init__()
        self.embed_dim = neox_args.hidden_size
        self.num_heads = neox_args.num_attention_heads
        
        self.gate_fn = nn.functional.silu

        self.q_proj = mpu.ColumnParallelLinear(neox_args=neox_args,
                                               input_size=self.embed_dim,
                                               output_size=self.embed_dim//2,
                                               bias=False,
                                               gather_output=True,
                                               init_method=init_method,
                                               skip_bias_add=not False)
    
        
        self.k_gate =  mpu.ColumnParallelLinear(neox_args=neox_args,
                                               input_size=self.embed_dim,
                                               output_size=self.embed_dim//2,
                                               bias=False,
                                               gather_output=True,
                                               init_method=init_method,
                                               skip_bias_add=not False)

        self.v_proj = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            gather_output=True,
            init_method=init_method,
            skip_bias_add=not False,
            bias=False,
        )
        
        self.g_proj = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            gather_output=True,
            init_method=init_method,
            skip_bias_add=not True,
            bias=True,
        )
        self.out_proj = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            input_is_parallel=False,
            init_method=output_layer_init_method,
            skip_bias_add=not False,
            bias=False,
            parallel_output=False,
        )

        self.head_dim = self.embed_dim // self.num_heads
        self.group_norm = LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False)
        
        self.aug_balance = nn.Parameter(0.0 * torch.zeros(self.embed_dim//2))

        self.d_conv = 4
        self.conv1d = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            bias=False,
            kernel_size=self.d_conv,
            groups=self.embed_dim,
            padding=self.d_conv - 1,
            # **factory_kwargs,
        )

    def forward(self, x, hidden_states=None):
        x = x.transpose(0, 1).contiguous()

        ##### short convolution #####
        x = rearrange(x, 'b l d -> b d l').contiguous()
        x = causal_conv1d_fn(
                x=x,
                weight=einops.rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias.to(self.precision)
                if self.conv1d.bias is not None
                else self.conv1d.bias,
                activation="silu",
            )
        x = rearrange(x, 'b d l -> b l d').contiguous()

        q, _ = self.q_proj(x)
        k_gate, _ = self.k_gate(x)
        k = 1
        v, _ = self.v_proj(x)
        g, _ = self.g_proj(x)

        output, new_hidden_states = self.meta_linear_attention(q, k, v, k_gate, hidden_states=hidden_states)
        output = self.gate_fn(g) * output
        output, _ = self.out_proj(output)
        output = output.transpose(0, 1)
        return output, new_hidden_states
    
    def meta_linear_attention(self, q, k, v, gk, normalizer=16, hidden_states=None):
        ##### remove key #####
        gk = F.logsigmoid(gk) / normalizer
        k = 1 - torch.exp(gk)
        
        q = rearrange(q, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        k = rearrange(k, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        v = rearrange(v, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        gk = rearrange(gk, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        aug_balance = rearrange(self.aug_balance, '(h d) -> h d', h = self.num_heads).contiguous()
        
        if self.training:
            o, new_hidden_states = fused_chunk_gla(q, k, v, gk, initial_state=hidden_states, output_final_state=True)
        else:
            o, new_hidden_states = fused_recurrent_gla(q, k, v, gk, initial_state=hidden_states, output_final_state=True)
        
        ##### self augmentation #####
        augk = torch.einsum('bhld,hd->bhld', k, aug_balance)
        aug_w = torch.einsum('bhld,bhld->bhl', q, augk)
        o = o + F.sigmoid(aug_w.unsqueeze(-1) * v)
            
        o = self.group_norm(o)
        o = rearrange(o, 'b h l d -> b l (h d)')
        return o, new_hidden_states


class ParallelMetaLALayer_selfaug(nn.Module):
    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        layer_number,
        use_cache=False
    ):
        super().__init__()
        
        assert not use_cache, "[MetaLA]: use_cache conflicts with training mode!"
        self.neox_args = neox_args
        self.layer_number = layer_number
        
        norm, eps = get_norm(neox_args)
        self.input_layernorm = norm(neox_args.hidden_size, eps=eps)
        self.post_attention_layernorm = norm(neox_args.hidden_size, eps=eps)
        self.hidden_dropout = neox_args.hidden_dropout

        self.attention = ParallelMetaLA_Attention_selfaug(neox_args, 
                                                             init_method, 
                                                             output_layer_init_method)
        
        self.mlp = LLaMAParallelMLP(
            neox_args=neox_args,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
        )
        
    def forward(self, x, attention_mask, layer_past=None):
        residual = x  # (l, b, d)
        
        moe_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        attention_output, _ = self.attention(self.input_layernorm(x))
        
        with torch.enable_grad():
            attention_output = (
                        torch.nn.functional.dropout(
                            attention_output,
                            p=self.hidden_dropout,
                            training=self.training,
                        )
                        + residual
                    )
            
        layernorm_output = self.post_attention_layernorm(attention_output)
        
        mlp_output, _ = self.mlp(layernorm_output)
        
        with torch.enable_grad():
            output = mlp_output + attention_output
            
        return output, moe_loss

  
class ParallelMetaLALayer_selfaugPipe(ParallelMetaLALayer_selfaug):
    """Extends ParallelMetaLALayer to forward attention_mask through the pipeline."""

    def forward(self, args):
        assert (
            len(args) == 2
        ), "ParallelMetaLALayer_selfaugPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        # we are returning just [hidden_states, mask]
        output, moe_loss = super().forward(hidden_states, attention_mask)
        # auxiliary output
        self.last_moe_loss = moe_loss
        return output, attention_mask