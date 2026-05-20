from turtle import forward

import einx
import torch
from torch import nn
from einops import einsum, rearrange
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
        self.register_buffer(
            "_freq_cis_cache",
            RotaryPositionEmbedding._init_cache(max_seq_len, d_k, theta), persistent=False
        )

    @staticmethod 
    def _init_cache(max_seq_len, d_k, theta ):
        d = torch.arange(0, d_k, 2)/d_k
        freqs = theta** - d
        t = torch.arange(0, max_seq_len)
        
        freqs = einsum(t, freqs, "t,d-> t d") 
        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))
        
    def forward(self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq d"]:
        """
        Apply RoPE to input tensor.

        Args:
            x:
                Input tensor of shape (..., seq_len, d_k).
                It may have arbitrary batch dimensions.

            token_positions:
                Tensor of shape (..., seq_len), specifying the token positions
                along the sequence dimension.

        Returns:
            Tensor of the same shape as x: (..., seq_len, d_k).
        """
        x1, x2 = rearrange(x, '... (half_d xy) -> xy ... half_d', xy=2)

        # Standard
        # cos, sin = self._freq_cis_cache[:, pos_ids, :]

        # einx
        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._freq_cis_cache, pos_ids)

        # 2D rotation matrix applied to pairs in x
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()
        return result
    
    def extra_repr(self):
        return f"context_length={self._freq_cis_cache.shape[0]}, dim/2={self._freq_cis_cache.shape[1]}"