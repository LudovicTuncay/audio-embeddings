import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomProjectionQuantizer(nn.Module):
    """Vector quantization using a projection and a randomly initialised codebook
    this is useful for models like BEST-RQ for instance.

    The output is the indices of the closest code in the codebook for each
    time step of the input.

    ref: https://arxiv.org/pdf/2202.01855

    Arguments
    ---------
    input_dim: int
        Input dimension (channels).
    cb_dim: int
        Size of each code in the codebook.
    cb_vocab: int
        Number of codes in the codebook

    Example
    -------
    >>> quantiser = RandomProjectionQuantizer(16, 16, 32)
    >>> inputs = torch.rand(10, 12, 16)
    >>> output = quantiser(inputs)
    >>> output.shape
    torch.Size([10, 12])
    """

    def __init__(self, input_dim, cb_dim, cb_vocab):
        super().__init__()

        self.input_dim = input_dim
        self.cb_dim = cb_dim
        self.cb_vocab = cb_vocab

        # Section 3.1 "projection matrix A use Xavier initialization"
        P_init = torch.empty((input_dim, cb_dim))
        self.register_buffer("P", nn.init.xavier_uniform_(P_init))

        # normalize random matrix for codebook
        self.register_buffer("CB", F.normalize(torch.randn(cb_vocab, cb_dim)))

    def forward(self, x):
        """Forward the latent vector to obtain a quantised output"""

        x = F.normalize(x @ self.P, dim=-1)
        # since both x and CB are normalized, we can just take the argmax of the dot product
        return F.linear(x, self.CB).argmax(dim=-1)
