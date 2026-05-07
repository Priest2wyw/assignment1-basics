import torch
from einops import einsum, rearrange

batch_size = 3
sequence_length = 4
d_model = 3

# basic implementation
D = torch.rand(batch_size, sequence_length, d_model)
A = torch.rand(d_model, d_model)
Y_torch = D @ A.T  # batch seq seq  

## Einsum is self-documeting and robust
Y_ein = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

assert torch.allclose(Y_torch, Y_ein)
Y_ein_sum = einsum(D, A, "... d_in, d_out d_in -> ... d_out")


### BROADCASTED OPERATIONS 
images = torch.randn(64, 128, 128, 3) # (batch_size, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

## Reshape and multiply
dim_value = rearrange(dim_by,    "dim_value              -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel   ")
dimmed_images = images_rearr * dim_value

# or in one go:
dimmed_images_enisum = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)

assert torch.allclose(dimmed_images, dimmed_images_enisum)


### ---- pixel mixing with einops.rearrange----
channels_last = torch.randn(64, 32, 32, 3)
B = torch.randn(32*32, 32*32)

# rerange an image tensor for mixing across all pixels
channels_last_flat = channels_last.view(
    -1, channels_last.size(1)* channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)

## -------- torch with jaxtyping -----------
from jaxtyping import Float
import torch

def transform(
    x: Float[torch.Tensor, "batch pixel channel"],
    B: Float[torch.Tensor, "out_pixel in_pixel"],
) -> Float[torch.Tensor, "batch out_pixel channel"]:
    return torch.einsum('bpc,qp->bqc', x, B)