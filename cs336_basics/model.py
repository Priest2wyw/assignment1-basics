import logging
import einx
import torch
from torch import nn
from einops import einsum, rearrange, reduce
from torch import Tensor
from jaxtyping import Float, Bool, Int

logger = logging.getLogger(__name__)

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
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = (2/(in_features + out_features))**0.5 # std 
        nn.init.trunc_normal_(self.weight ,mean=0, std=std, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in-> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """Construct an embedding module. This function should accept the following parameters

            num_embeddings: int  Size of the vocabulary
            embedding_dim: int  Dimension of the embedding vectors, i.e., 𝑑model
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim), 
                device=device, dtype=dtype
                )
            )
        torch.nn.init.trunc_normal_(self.weight, mean = 0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs"""
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """Construct the RMSNorm module. This function should accept the following parameters:

            d_model: int  Hidden dimension of the model
            eps: float = 1e-5  Epsilon value for numerical stability
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d = d_model
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape 
        (batch_size, sequence_length, d_model) and return a tensor of the same shape."""
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # compute rmsnorm
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True)+self.eps)
        results = torch.mul(x, self.weight)
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
        self.w1 = Linear(d_model, d_ff   )
        self.w2 = Linear(d_ff   , d_model)
        self.w3 = Linear(d_model, d_ff   )
        
    def forward(self, in_features):
        gate = self.w1(in_features)
        value = self.w3(in_features) 
        hidden = SiLU(gate) * value

        output = self.w2(hidden)
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
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        theta: float=10_000.0,
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

def softmax(x: Tensor, dim:int)-> Tensor:
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
    attention_matix = einsum(Q,K,"... q d_k, ... k d_k -> ... q k")/ (d_k**(0.5))
    # add mask
    masked_atten = attention_matix.masked_fill(~mask, -torch.inf)
    atten_prob = softmax(masked_atten, dim=-1)
    # value
    atten_score =  einsum(atten_prob, V, " ... q k, ... k d_v -> ... q d_v")
    return atten_score
    
    
class CauseMultheadSelfAttention(nn.Module):
    """
    Args:
        d_model: dimon of model
        num_heads: number of heads
        positional_encoder: position encoder ,default use rope 
    Output:
        output: Float[Tensor, " ... sequence_length d_model"]:
    """
    
    def __init__(self, d_model: int, 
                 num_heads: int,
                 positional_encoder: RotaryPositionEmbedding|None=None):
        super().__init__()
        assert d_model % num_heads == 0
        # ASSUME: d_k = d_v = d_model/num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model/num_heads
        self.d_v = d_model/num_heads

        self.q_proj = Linear(d_model, int(num_heads*self.d_k) )
        self.k_proj = Linear(d_model, int(num_heads*self.d_k) )
        self.v_proj = Linear(d_model, int(num_heads*self.d_v) )
        self.output_proj = Linear(int(num_heads*self.d_v), d_model)
        self.positional_encoder = positional_encoder

    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"], token_positions=None):
        *b, seq_len, d_model = in_features.size()
        Q = self.q_proj(in_features)# ... seq_len num_head*d_k
        K = self.k_proj(in_features)# ... seq_len num_head*d_k
        V = self.v_proj(in_features)# ... seq_len num_head*d_v

        Q = rearrange(Q, "... seq (num_head d_k) -> ... num_head seq d_k", num_head=self.num_heads)
        K = rearrange(K, "... seq (num_head d_k) -> ... num_head seq d_k", num_head=self.num_heads)
        V = rearrange(V, "... seq (num_head d_v) -> ... num_head seq d_v", num_head=self.num_heads)

        # add token positions
        if self.positional_encoder is not None:
            if token_positions is None:
                token_positions = einx.rearrange("seq -> b... seq", torch.arange(seq_len, device=in_features.device), b=[1] * len(b))

            # TODO: why do this? explain this 
            # Duplicate token positions for each head
            token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
            Q = self.positional_encoder(Q, token_positions)
            K = self.positional_encoder(K, token_positions)
            
        # build mask matrix
        # torch.triu(torch.ones([3,3]), diagonal=1)
        # tensor([[0., 1., 1.],
        #         [0., 0., 1.],
        #         [0., 0., 0.]])
        masked_matrix = torch.triu(torch.ones([seq_len, seq_len]), diagonal=1) < 0.5
        atten_out = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=masked_matrix) # ... num_head seq d_v
        
        # W_O: d_model*(num_head d_v)
        atten_out = rearrange(atten_out, "... num_head seq d_v-> ... seq (num_head d_v)")
        output = self.output_proj(atten_out)
        return output 

class TransformerBlock(nn.Module):
    """
        Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, position_encoder: RotaryPositionEmbedding):
        super().__init__()
        self.attn = CauseMultheadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            positional_encoder=position_encoder
            )
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff)
    
    def forward(self, x:Float[Tensor, "batch sequence_length d_model"])->Float[Tensor, "batch sequence_length d_model"]:
        ln1_out = self.ln1(x) # batch sequence_length d_model
        attention_out = self.attn(ln1_out)
        pre_norm_attantion = x + attention_out
        
        ln2_out = self.ln2(pre_norm_attantion)
        ffn_out = self.ffn(ln2_out)
        block_out = ffn_out + pre_norm_attantion
        
        return block_out
        

class BasicsTransformerLM(nn.Module):
    """A Transformer language model.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta: float
            The theta value for the RoPE positional encoding.

    Returns:
        FloatTensor of shape (batch size, sequence_length, vocab_size) with the
        predicted unnormalized next-word distribution for each token.
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = int(d_model/num_heads)
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.context_length = context_length

        self.position_encoder = RotaryPositionEmbedding(
            d_k = self.d_k, 
            max_seq_len=context_length, 
            theta = rope_theta
            )
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff = d_ff,
                position_encoder=self.position_encoder
            ) 
            for i in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)
       
        para_numb = self.get_para_number()/ 10**6 
        logger.error(f"number of paras is {para_numb:,.3f}M, if single-precision floating poinit need {4* para_numb:,.3f}M ")

    def forward(self, in_indics:Int[Tensor, "batch_size sequence_length"])->Float[Tensor, "batch_size sequence_length vocab_size"]:
        embds = self.token_embeddings(in_indics)
        for layer in self.layers:
            embds = layer(embds)
        embds = self.ln_final(embds)
        embds = self.lm_head(embds)
        return embds

    def get_para_number(self, no_embd = True):
        all_number = sum([b.numel() for b in self.parameters()])

        if no_embd:
            all_number -= self.lm_head.weight.numel()
        return all_number

def cross_entropy(input:Float[Tensor, "batch_size vocab_size"], 
                  target:Float[Tensor, "batch_size vocab_size"]):
    """
    Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        output (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    """
    tips for numerical stability:
    1. softmax(x) = softmax(x-x_{max})
    2. logsoftmax(x) = log \frac{e^{x_i}}{\sum e^{x_i}} = x_i- log \sum e^{x_i}
    """
    input_subtract_max= input - reduce(input, "... v-> ... 1", "max")# b s v
    log_prob = input_subtract_max - input_subtract_max.logsumexp(dim=-1, keepdim=True)
    neg_log_prob = - log_prob
    
    simple_prob = torch.gather(neg_log_prob, dim = -1, index =target.unsqueeze(dim=-1))
    
    return simple_prob.mean()
    
        
if __name__ == "__main__":
    vocab_size=50257
    context_length= 1024
    num_layers= 48
    d_model= 1600
    num_heads= 25
    d_ff= 4288
    gpt_2_xl = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta=10000)