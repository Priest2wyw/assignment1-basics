from re import S
from turtle import forward

import einx
import torch
from torch import nn
from einops import einsum, rearrange, reduce
from torch import Tensor
from jaxtyping import Float, Bool, Int


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
            transformation module. This function should accept the following parameters:

            in_features: int  final dimension of the input
            out_features: int  final dimension of the output
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
            def forward(self, x: torch.Tensor) -> torch.Tensor
        """
        super().__init__()
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = (2/(in_features + out_features))**0.5 # std 
        nn.init.trunc_normal_(self.W ,mean=0, std=std, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in-> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """Construct an embedding module. This function should accept the following parameters

            num_embeddings: int  Size of the vocabulary
            embedding_dim: int  Dimension of the embedding vectors, i.e., 𝑑model
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.W = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(self.W, mean = 0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs"""
        return self.W[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """Construct the RMSNorm module. This function should accept the following parameters:

            d_model: int  Hidden dimension of the model
            eps: float = 1e-5  Epsilon value for numerical stability
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d = d_model
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape 
        (batch_size, sequence_length, d_model) and return a tensor of the same shape."""
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # compute rmsnorm
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True)+self.eps)
        results = torch.mul(x, self.g)
        results = torch.mul(results, 1/rms)
        
        results.to(in_dtype)
        return results

def SiLU(x):
    return x * torch.sigmoid(x)

class FeedForwardNetwork(nn.Module):
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Parameter:
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3


    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model))

        self._init_parameter()

    def _init_parameter(self):
        nn.init.xavier_uniform_(self.W1) 
        nn.init.xavier_uniform_(self.W2) 
        nn.init.xavier_uniform_(self.W3) 
        
    def forward(self, in_features):
        gate = einsum(in_features, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        value = einsum(in_features, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        hidden = SiLU(gate) * value

        output = einsum( hidden, self.W2, " ... d_ff,d_model d_ff -> ... d_model")
        return output

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Positional Embedding, RoPE.

    Args:
        theta: Θ value for RoPE.
        d_k: Dimension of query and key vectors.
        max_seq_len: Maximum sequence length that will be input.
        device: Device to store buffers on.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE")

        dim_index = torch.arange(0, d_k, 2, device=device).float()
        denominator = 1 / (theta ** (dim_index / d_k))
        
        position = torch.arange(max_seq_len, device=device).float()
        angles = torch.outer(position, denominator)
        
        self.register_buffer("sin_cache", torch.sin(angles), persistent= False)
        self.register_buffer("cos_cache", torch.cos(angles), persistent= False)
        
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: 
        x_even = x[..., 0::2]
        x_odd = x [..., 1::2]
        
        cos = self.cos_cache[token_positions].to(x.dtype)
        sin = self.sin_cache[token_positions].to(x.dtype)

        out_even = x_even*cos - x_odd*sin
        out_odd = x_even*sin + x_odd*cos

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd

        return out

def softmax(x: torch.tensor, dim:int)-> torch.tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x = x - x_max
    
    x_exp = torch.exp(x)
    sum_exp = torch.sum(x_exp, dim=dim, keepdim=True)
    soft_max = x_exp /  sum_exp
    
    return soft_max

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    attention_matix = einsum(Q,K,"... q d_k, ... k d_k -> ... q k")/ (d_k)**(0.5)
    # add mask
    masked_atten = attention_matix.masked_fill(~mask, -torch.inf)
    atten_prob = softmax(masked_atten, dim=-1)
    # value
    atten_score =  einsum(atten_prob, V, " ... q k, ... k d_v -> ... q d_v")
    return atten_score
    
    
    