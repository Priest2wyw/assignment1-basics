import torch
from torch import nn
from einops import einsum


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

        