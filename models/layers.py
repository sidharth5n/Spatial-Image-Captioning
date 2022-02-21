from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from misc.utils import clones

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class RMSNorm(nn.Module):
    def __init__(self, d_model, p = -1., eps = 1e-8, bias = False):
        """
        Root Mean Square Layer Normalization
        https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py

        Parameters
        ----------
        d_model : int
                  Model size
        p       : float, optional
                  Partial RMSNorm, valid value [0, 1]. Default value is -1.0 (disabled)
        eps     : float, optional
                  Epsilon value. Default value is 1e-8.
        bias    : bool, optional
                  Whether to use bias term for RMSNorm. Default is False.
                  Disabled by default because RMSNorm doesn't enforce re-centering
                  invariance.
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d_model
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(self.d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(self.d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout, norm = 'layer'):
        """
        Parameters
        ----------
        size    : int, list or torch.Size
                  Expected shape of input
        dropout : float
                  Dropout probability
        """
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size) if norm == 'layer' else RMSNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    d_model -> d_ff -> ReLU -> dropout -> d_model
    """
    def __init__(self, d_model, d_ff, dropout = 0.1, activation = 'RELU'):
        """
        Parameters
        ----------
        d_model : int
                  Size of input features
        d_ff    : int
                  Intermediate size
        dropout : float
                  Dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()
        self.fc = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.ReLU() if activation == 'RELU' else GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.fc(x)

def attention(query, key, value, mask = None, dropout = None):
    """
    Computes Scaled Dot Product Attention.

    B - Batch Size
    H - Number of attention heads
    Q - Query length
    K - Key length
    V - Value length
    d - Feature dimension

    Parameters
    ----------
    query   : torch.tensor of shape (B, H, Q, d)
    key     : torch.tensor of shape (B, H, K, d)
    value   : torch.tensor of shape (B, H, V, d)
    mask    : torch.tensor of shape (B, 1, Q, K), optional
              Mask to be applied. Default is None.
    dropout : torch.nn.Dropout, optional
              Dropout layer. Default is None.

    Returns
    -------
    out     : torch.tensor of shape (B, H, Q, d)
              Attended feature vector
    p_attn  : torch.tensor of shape (B, H, Q, K)
              Attention weights
    """
    d = query.shape[-1]
    # Compute dot product similarity b/w query and key, (B,H,Q,K)
    scores = torch.einsum("bhqd,bhkd->bhqk", [query, key])
    # Scale dot product similarity
    scores = scores / math.sqrt(d)
    # Mask scores if mask is available
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # Softmax normalize the scores and apply dropout if available, (B,H,Q,K)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # Compute attention weighted features, (B,H,Q,d)
    out = torch.einsum("bhqk,bhkd->bhqd", [p_attn, value])
    return out, p_attn

class MultiHeadedAttention(nn.Module):
    """
    Computes multiple attentions by linearly projecting queries, keys and values.
    """
    def __init__(self, heads, d_model, dropout = 0.1):
        """
        Parameters
        ----------
        heads   : int
                  Number of attention heads
        d_model : int
                  Size of input features
        dropout : float, optional
                  Dropout probability. Default is 0.1.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // heads
        self.heads = heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value, mask = None):
        """
        Implements Figure 2

        B - Batch Size
        H - Number of attention heads
        Q - Query length
        K - Key length
        V - Value length
        S - Sequence length (One of Q, K, V)
        D - Feature dimension

        Parameters
        ----------
        query   : torch.tensor of shape (B, Q, D)
        key     : torch.tensor of shape (B, K, D)
        value   : torch.tensor of shape (B, V, D)
        mask    : torch.tensor of shape (B, Q, K), optional
                  Mask to be applied. Default is None.

        Returns
        -------
        out     : torch.tensor of shape (B, Q, D)
                  Attended feature vector for each query
        """
        if mask is not None: # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.shape(0)
        # Linear projection creating multiple heads, (B,S,D)->(B,S,D)->(B,S,H,d)->(B,H,S,d)
        query, key, value = [l(x).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # Applying attention on all the projected vectors in batch, (B,H,Q,d), (B,H,Q,K)
        x, self.attn = attention(query, key, value, mask, self.dropout)
        # Concatenate all the attention heads and apply linear layer, (B,H,Q,d)->(B,Q,H,d)->(B,Q,H*d=D)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.heads * self.d_k)
        # (B,Q,H*d_k=d)->(B,Q,d)
        out = self.linears[-1](x)
        return out

class XLinearMultiHeadedAttention(nn.Module):
    """
    Implements X-Linear attention block. Refer Fig. 1(b) Pan et al. "X-Linear
    Attention Networks for Image Captioning"
    """

    def __init__(self, heads, d_model, mid_dim = None, dropout = 0.1):
        """
        Parameters
        ----------
        heads     : int
                    Number of attention heads
        d_model   : int
                    Size of input features
        mid_dim   : int, optional
                    Size of embedding. Default is None.
        dropout   : float, optional
                    Dropout probability. Default is 0.1.
        """
        super(XLinearMultiHeadedAttention, self).__init__()
        assert (d_model % heads) == 0
        self.d_k = d_model // heads # We assume d_v always equals d_k
        self.heads = heads
        linear = nn.Sequential(nn.Linear(d_model, d_model),
                               nn.CELU(),
                               torch.nn.GroupNorm(self.heads, d_model))
        self.linears = clones(linear, 4)
        self.attention = Attention(self.d_k, mid_dim, dropout)

    def forward(self, query, key, value, mask = None):
        """
        Performs bilinear pooling and computes spatial and channel attended
        features for each query.

        Parameters
        ----------
        query : torch.tensor of shape (B, Q, E)
        key   : torch.tensor of shape (B, K, E)
        value : torch.tensor of shape (B, K, E)
        mask  : torch.tensor of shape (B, Q, K)
                Mask to be applied. Default is None.

        Returns
        -------
        out   : torch.tensor of shape (B, Q, E)
                Bilinear pooled spatial and channel attended feature for each query.
        """
        if mask is not None: # Apply same mask to all heads (B,Q,K)->(B,1,Q,K)
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # Convert to multi-head, (B,Q,E)->(B,H,Q,D)    (B,K,E)->(B,H,K,D)
        q1, k, q2, v = [l(x.view(-1, x.size(-1))).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
                        for l, x in zip(self.linears, (query, key, query, value))]
        # Compute spatial and channel attended feature, (B,H,Q,D)
        x = self.attention(q1, k, q2, v, mask)
        # Combine all heads, (B,H,Q,D)->(B,Q,H,D)->(B,Q,H*D=E)
        out = x.transpose(1, 2).contiguous().view(nbatches, -1, self.heads * self.d_k)
        return out

class Attention(nn.Module):
    """
    Computes spatial and channel attended feature.
    """
    def __init__(self, embed_dim, mid_dim, dropout):
        """
        Parameters
        ----------
        embed_dim : int
                    Size of input feature
        dropout   : float
                    Dropout probability
        mid_dim   : int, optional
                    Size of embedding. Default is embed_dim/2.
        """
        super(Attention, self).__init__()
        if mid_dim is None:
            mid_dim = embed_dim // 2
        self.embed = nn.Sequential(nn.Linear(embed_dim, mid_dim),
                                   nn.ReLU(),
                                   nn.Dropout(dropout))
        self.spatial = nn.Linear(mid_dim, 1)
        self.squeeze = nn.Sequential(nn.Linear(mid_dim, embed_dim),
                                     nn.Sigmoid())

    def forward(self, query1, key, query2, value, mask):
        """
        Parameters
        ----------
        query1      : torch.tensor of shape (B, H, Q, D)
                      Query for computation of key map.
        key         : torch.tensor of shape (B, H, K, D)
        query2      : torch.tensor of shape (B, H, Q, D)
                      Query for computation of value map.
        value       : torch.tensor of shape (B, H, K, D)
        mask        : torch.tensor of shape (B, H, Q, K)
                      Mask to be applied.

        Returns
        -------
        channel_att : torch.tensor of shape (B, H, Q, D)
                      Spatial and channel attended features for each query.
        """
        # (B,H,Q,1,D)*(B,H,1,K,D)->(B,H,Q,K,D)
        key_map = torch.einsum("bhqd,bhkd->bhqkd", [query1, key])
        # (B,H,Q,K,D)->(B,H,Q,K,D/2)
        key_map = self.embed(key_map)
        # (B,H,Q,K,D/2)->(B,H,Q,K,1)->(B,H,Q,K)
        alpha_spatial = self.spatial(key_map).squeeze(-1)
        if mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(mask == 0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)

        if mask is not None: # (B,H,Q,K,D/2)->(B,H,Q,D/2)
            key_map_pool = torch.sum(key_map * mask.unsqueeze(-1), -2) / torch.sum(mask.unsqueeze(-1), -2)
        else:
            key_map_pool = key_map.mean(-2)
        # (B,H,Q,D/2)->(B,H,Q,D)
        alpha_channel = self.squeeze(key_map_pool)

        # (B,H,Q,1,D)*(B,H,1,K,D)->(B,H,Q,K,D)
        val_map = torch.einsum("bhqd,bhkd->bhqkd", [query2, value])
        # (B,H,Q,K,1)*(B,H,1,K,D)->(B,H,Q,D)
        spatial_att = torch.einsum("bhqk,bhqkd->bhqd", [alpha_spatial, val_map])
        # (B,H,Q,D)*(B,H,Q,D)
        channel_att = spatial_att * alpha_channel

        return channel_att

class EncoderLayer(nn.Module):
    """
    Implements one layer of the encoder consisting of self attention and
    position wise feed forward. Each sub-module is wrapped around layer norm and
    residual connection.
    """
    def __init__(self, size, self_attn, feed_forward, norm, dropout):
        """
        Parameters
        ----------
        size         : int
                       Feature size
        self_attn    : Attention layer
        feed_forward : Position wise feed forward layer
        dropout      : float
                       Dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, norm), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Follow Figure 1 (left) for connections.

        Parameters
        ----------
        x    : torch.tensor of shape (B, P, E)
               Input sequence
        mask : torch.tensor of shape (B, P, P)
               Input sequence mask

        Returns
        -------
        x    : torch.tensor of shape (B, P, E)
               Output sequence
        """
        # Compute self attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # Compute position-wise feed forward output
        x = self.sublayer[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, norm, N):
        """
        Parameters
        ----------
        layer : EncoderLayer
        N     : int
                No. of encoder layers
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size) if norm == 'layer' else RMSNorm(layer.size)

    def forward(self, x, mask = None):
        """
        Pass the input and mask through each layer sequentially.

        Parameters
        ----------
        x    : torch.tensor of shape (B, P, E)
               Embedding of the input sequence.
        mask : torch.tensor of shape (B, P), optional
               Input sequence mask. Default is None.

        Returns
        -------
        x    : torch.tensor of shape (B, P, E)
               Encoded sequence
        """
        # Pass the data sequentially through all the encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        # Layer normalize the final output
        x = self.norm(x)
        return x

class DecoderLayer(nn.Module):
    """
    Implements one layer of the decoder consisting of local attention, self
    attention, cross attention and position wise feed forward. Each sub-module
    (except local attention) is wrapped around layer norm and residual connection.
    """
    def __init__(self, size, self_attn, cross_attn, feed_forward, norm, dropout = 0.1):
        """
        Parameters
        ----------
        size         : int
                       Size of input
        self_attn    : Self attention layer
        cross_attn   : Cross attention layer
        feed_forward : PositionwiseFeedForward layer
        local_attn   : Local attention layer, optional
        dropout      : float, optional
                       Dropout probability. Default is 0.1.
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, norm), 3)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        """
        Computes one layer of decoder.

        Parameters
        ----------
        x          : torch.tensor of shape (B, T+1, H)
                     Embedding of the target sequence
        enc_out    : torch.tensor of shape (B, P, E)
                     Encoded input sequence
        src_mask   : torch.tensor of shape (B, P)
                     Source mask
        tgt_mask   : torch.tensor of shape (B, T+1, T+1)
                     Target mask

        Returns
        -------
        x          : torch.tensor of shape (B, T+1, H)
                     Output sequence
        """
        m = enc_out
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask = tgt_mask))
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, m, m, mask = src_mask))
        x = self.sublayer[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, norm, N):
        """
        Parameters
        ----------
        layer : DecoderLayer
        N     : int
                No. of decoder layers
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size) if norm == 'layer' else RMSNorm(layer.size)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        """
        Parameters
        ----------
        x          : torch.tensor of shape (B, T+1, H)
                     Embedding of the target sequence
        enc_out    : torch.tensor of shape (B, P, E)
                     Encoded input sequence
        src_mask   : torch.tensor of shape (B, P, P)
                     Source mask
        tgt_mask   : torch.tensor of shape (B, T+1, T+1), optional
                     Target mask. Default is None.

        Returns
        -------
        x          : torch.tensor of shape (B, T+1, H)
                     Decoded target sequence
        """
        # Pass the data and encoder output sequentially through all the decoder layers
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        # Layer normalize the final output
        x = self.norm(x)
        return x

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Implement the PE function as given in the paper - Attention is All You Need.
    """
    def __init__(self, d_model, dropout, max_len = 5000):
        """
        Parameters
        ----------
        d_model : int
                  Size of input
        dropout : float
                  Dropout probability
        max_len : int, optional
                  Maximum length of the sequence. Default is 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input.

        Parameters
        ----------
        x : torch.tensor of shape (B, D)
            Input

        Returns
        -------
        x : torch.tensor of shape (B, D)
            Input with position encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SpatialPositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """
    def __init__(self, grids, d_model, dropout, learnable = True, learnable_type = 'gxg'):
        """
        Parameters
        ----------
        grids   : int
                  Number of grids if using grid vectors
        d_model : int
                  Size of input
        dropout : float
                  Dropout probability
        learnable : bool
                    Whether using learnmable embedding
        learnable_type : str
                         Type of learnable embedding. One of 'gxg', 'g2' or 'box-coord'
        """
        super(SpatialPositionalEncoding, self).__init__()
        assert int(grids ** 0.5) ** 2 == grids, "Grids must be a whole square"
        grids = int(grids ** 0.5)
        self.learnable = learnable
        self.dropout = nn.Dropout(p = dropout)
        buffer = False

        if learnable:
            assert learnable_type in ['gxg', 'g2', 'box-coord'], "learnable_type not available"
            self.learnable_type = False
            if learnable_type == 'gxg':
                print("Initializing grid based encoding")
                self.fc1 = nn.Linear(grids, d_model)
                self.fc2 = nn.Linear(grids, d_model)
                self.fc3 = nn.Linear(d_model, d_model)
                self.grids = grids
                buffer = True
                self.learnable_type = True
            elif learnable_type == 'g2':
                self.fc = nn.Linear(grids ** 2, d_model)
            else:
                self.fc = nn.Sequential(nn.Linear(4, d_model//2),
                                        nn.ReLU(),
                                        nn.Linear(d_model//2, d_model))
        else:
            # Compute the positional encodings once in log space.
            pe1 = torch.zeros(grids, d_model)
            pe2 = torch.zeros(grids, d_model)
            position = torch.arange(0, grids).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            pe1[:, 0::2] = torch.sin(position * div_term)
            pe1[:, 1::2] = torch.cos(position * div_term)
            pe2[:, 0::2] = torch.cos(position * div_term)
            pe2[:, 1::2] = torch.sin(position * div_term)
            self.register_buffer('pe1', pe1)
            self.register_buffer('pe2', pe2)
            buffer = True

        if buffer:
            rows = (torch.arange(grids) % grids).long()
            cols = (torch.arange(grids) // grids).long()
            self.register_buffer('rows', rows)
            self.register_buffer('cols', cols)

    def forward(self, img_feat, box_vec):
        """
        Adds positional encoding to the input.

        Parameters
        ----------
        img_feat : torch.tensor of shape (B, L, D)
                   Input
        box_vec  : torch.tensor of shape (B, L, G) or (B, L, 4)
                   Grid vectors or (x, y, w, h) of Bounding Boxes

        Returns
        -------
        img_feat : torch.tensor of shape (B, L, D)
                   Input with position encoding added.
        """
        if self.learnable:
            if self.learnable_type:
                # img_feat = img_feat + self.fc1(box_vec[...,self.rows])
                # img_feat = img_feat + self.fc2(box_vec[...,self.cols])
                nbatches = box_vec.shape[0]
                box_vec = box_vec.view(nbatches, -1, self.grids, self.grids)
                pe = self.fc1(box_vec.sum(-1)) + self.fc2(box_vec.sum(-2))
                img_feat = self.fc3(img_feat) + pe
            else:
                img_feat = img_feat + self.fc(box_vec)
        else:
            img_feat = img_feat + torch.einsum("blg,gd->bld", [box_vec[...,self.rows, pe1]])
            img_feat = img_feat + torch.einsum("blg,gd->bld", [box_vec[...,self.cols, pe2]])
        return self.dropout(img_feat)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        Parameters
        ----------
        encoder   : Encoder
        decoder   : Decoder
        src_embed : Source embedding layer
        tgt_embed : Target embedding layer including positional encoding if any.
        generator : Generator
                    Projects to vocab size and gives log-softmax output
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_pe, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.

        Parameters
        ----------
        src        : torch.tensor of shape (B, P, E)
                     Input sequence
        tgt        : torch.tensor of shape (B, T+1)
                     1-indexed target sequence including <START> but excluding
                     <END>.
        src_mask   : torch.tensor of shape (B, P, P)
                     Input sequence mask.
        tgt_mask   : torch.tensor of shape (B, T+1, T+1)
                     Target sequence mask.

        Returns
        -------
        dec_out    : torch.tensor of shape (B, T+1, H)
                     Decoder output
        """
        # Compute embedding of the source sequence and run the encoder, (B,P,E)->(B,P,E)
        enc_out = self.encode(src, src_pe, src_mask)
        # Compute embedding of the target seqeuence and run the decoder, (B,T+1,H)->(B,T+1,H)
        dec_out = self.decode(enc_out, src_mask, tgt, tgt_mask)
        return dec_out

    def encode(self, src, src_pe, src_mask):
        """
        Encodes the given input.

        Parameters
        ----------
        src      : torch.tensor of shape (B, P)
                   Input sequence
        src_mask : torch.tensor of shape (B, P, P)
                   Input sequence mask.

        Returns
        -------
        enc_out  : torch.tensor of shape (B, P, E)
                   Encoder output
        """
        # Compute embedding of the source sequence, (B,P)->(B,P,E)
        src_emb = self.src_embed(src, src_pe)
        # Encode the source sequence, (B,P,E)->(B,P,E)
        enc_out = self.encoder(src_emb, src_mask)
        return enc_out

    def decode(self, enc_out, src_mask, tgt, tgt_mask):
        """
        Decodes the target sequence.

        Parameters
        ----------
        enc_out    : torch.tensor of shape (B, P, E)
                     Encoder output
        src_mask   : torch.tensor of shape (B, P)
                     Input sequence mask
        tgt        : torch.tensor of shape (B, T+1)
                     1-indexed target sequence including <START> but excluding
                     <END>.
        tgt_mask   : torch.tensor of shape (B, T+1, T+1)
                     Target sequence mask
        """
        # Compute the embedding of the target sequence, (B,T+1)->(B,T+1,H)
        tgt_emb = self.tgt_embed(tgt)
        # Decode the target sequence, (B,T+1,H)->(B,T+1,H)
        dec_out = self.decoder(tgt_emb, enc_out, src_mask, tgt_mask)
        return dec_out

class Generator(nn.Module):
    """
    Projects input to vocabulary size and gives log-softmax output.
    """
    def __init__(self, d_model, vocab):
        """
        Parameters
        ----------
        d_model : int
                  Model output size
        vocab   : int
                  Vocabulary size
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.tensor of shape (*, d_model)

        Returns
        -------
        x : torch.tensor of shape (*, vocab)
        """
        y = F.log_softmax(self.proj(x), dim=-1)
        return y
