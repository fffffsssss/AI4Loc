import torch
from torch import nn
from einops import rearrange

import ailoc.transloc
import ailoc.common


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, seq_length):
        super(PositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim))  # 1x

    def forward(self, x):
        position_embeddings = self.position_embeddings
        return x + position_embeddings


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=True, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x, attn_mask):
        batch_size, seq_length, d_model = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_length, 3, self.num_heads, d_model // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (k @ q.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(attn_mask==1, -torch.finfo(attn.dtype).max)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_length, d_model)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs


class TransLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            mlp_dim,
            attn_dropout_rate=0.1,
            ff_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []

        # for _ in range(depth):
        #     layers.extend(
        #         [
        #             ailoc.transloc.Residual(
        #                 PreNormDrop(
        #                     dim=dim,
        #                     dropout_rate=input_dropout_rate,
        #                     fn=SelfAttention(dim=dim, heads=heads, dropout_rate=attn_dropout_rate),
        #                 )
        #             ),
        #             ailoc.transloc.Residual(
        #                 PreNorm(
        #                     dim=dim,
        #                     fn=FeedForward(dim=dim, hidden_dim=mlp_dim, dropout_rate=input_dropout_rate)
        #                 )
        #             ),
        #         ]
        #     )

        # without norm
        for _ in range(depth):
            layers.extend(
                [
                    ailoc.transloc.Residual(
                        fn=SelfAttention(dim=dim, heads=heads, dropout_rate=attn_dropout_rate),
                    ),
                    ailoc.transloc.Residual(
                        fn=FeedForward(dim=dim, hidden_dim=mlp_dim, dropout_rate=ff_dropout_rate)
                    ),
                ]
            )

        # self.net = IntermediateSequential(*layers, return_intermediate=False)
        self.net = nn.ModuleList(layers,)

    def forward(self, x, attn_mask):
        # return self.net(x, attn_mask)
        for layer in self.net:
            x = layer(x, attn_mask)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, seq_length, attn_length, c_input, patch_size, embedding_dim):
        super().__init__()

        self.seq_length = seq_length
        self.attn_length = attn_length
        self.c_input = c_input
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        # self.patch_embedding = nn.Conv2d(in_channels=self.c_input,
        #                                  out_channels=self.embedding_dim,
        #                                  kernel_size=self.patch_size,
        #                                  stride=self.patch_size)

        # self.position_embedding = PositionalEncoding(seq_length=self.seq_length,
        #                                              embedding_dim=self.embedding_dim)

        # self.reverse_embedding = nn.Conv2d(in_channels=self.embedding_dim,
        #                                    out_channels=self.c_input*patch_size ** 2,
        #                                    kernel_size=1,
        #                                    stride=1)

    def forward(self, x):
        batch_size, context_size, feature_size, height, width = x.size()
        # assert context_size == self.seq_length, "context size is not equal to seq_length"
        assert feature_size == self.c_input, "feature size is not equal to c_input"
        assert height == width, "height is not equal to width"
        assert height % self.patch_size == 0, "height is not divisible by patch_size"
        self.patch_num_h = height // self.patch_size

        # # Conv2D version
        # x = rearrange(x, 'b c f h w -> (b c) f h w')
        # x = self.patch_embedding(x)  # (batch_size*seq_length, embedding_dim, patch_num_h, patch_num_w)
        # x = rearrange(x, '(b c) d ph pw -> (b ph pw) c d', b=batch_size,)
        # # x = self.position_embedding(x)

        # # simple partition, embedding dim = patch_size ** 2
        # x = rearrange(x, 'b c f (pn1 ps1) (pn2 ps2) -> b c f (pn1 pn2) ps1 ps2',
        #               ps1=self.patch_size, ps2=self.patch_size)
        # x = rearrange(x, 'b c f pn ps1 ps2 -> (b f pn) c (ps1 ps2)')

        # simple partition, embedding dim = patch_size ** 2 * feature_size
        x = rearrange(x, 'b c f (pn1 ps1) (pn2 ps2) -> b c f (pn1 pn2) ps1 ps2',
                      ps1=self.patch_size, ps2=self.patch_size)
        x = rearrange(x, 'b c f pn ps1 ps2 -> (b pn) c (f ps1 ps2)')

        # attention mask for each frame, can only see attn_length//2 frames around it
        neighbour = self.attn_length // 2
        attn_mask = torch.ones((context_size, context_size), device=x.device)
        for i in range(context_size):
            attn_mask[i, max(0, i - neighbour):min(context_size, i + neighbour + 1)] = 0
            attn_mask[max(0, i - neighbour):min(context_size, i + neighbour + 1), i] = 0

        return x, attn_mask

    def backward(self, x):
        # x = rearrange(x, '(b ph pw) c d -> (b c) d ph pw', ph=self.patch_num_h, pw=self.patch_num_h)
        # x = self.reverse_embedding(x)  # (batch_size*seq_length, feature_size*patch_size**2, patch_num_h, patch_num_h)
        # x = rearrange(x, '(b c) (f ps1 ps2) ph pw -> b c f (ps1 ph) (ps2 pw)', c=self.seq_length, f=self.c_input,
        #               ps1=self.patch_size, ps2=self.patch_size)

        # # simple partition, embedding dim = patch_size ** 2
        # x = rearrange(x, '(b f pn) c (ps1 ps2) -> b c f pn ps1 ps2',
        #               f=self.c_input, pn=self.patch_num_h**2, ps1=self.patch_size, ps2=self.patch_size)
        # x = rearrange(x, 'b c f (pn1 pn2) ps1 ps2 -> b c f (pn1 ps1) (pn2 ps2)',
        #               pn1=self.patch_num_h, pn2=self.patch_num_h, ps1=self.patch_size, ps2=self.patch_size)

        # simple partition, embedding dim = patch_size ** 2 * feature_size
        x = rearrange(x, '(b pn) c (f ps1 ps2) -> b c f pn ps1 ps2',
                      f=self.c_input, pn=self.patch_num_h**2, ps1=self.patch_size, ps2=self.patch_size)
        x = rearrange(x, 'b c f (pn1 pn2) ps1 ps2 -> b c f (pn1 ps1) (pn2 ps2)',
                      pn1=self.patch_num_h, pn2=self.patch_num_h, ps1=self.patch_size, ps2=self.patch_size)

        return x


class TransformerBlock(nn.Module):
    """
    Transformer block, the images features are split into patches, however, the MSA is applied
    to the feature patches along the context dimension, not the spatial dimensions.
    """

    def __init__(
            self,
            seq_length=10,
            attn_length=3,
            c_input=48,
            patch_size=1,
            embedding_dim=48,  # feature patch size
            num_layers=2,
            num_heads=8,
            mlp_dim=48*4,  # embedding_dim * 4
            dropout_rate=0.1,
            context_dropout=0.5,
    ):
        super(TransformerBlock, self).__init__()
        self.seq_length = seq_length
        self.attn_length = attn_length
        self.c_input = c_input
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.embedding_layer = EmbeddingLayer(self.seq_length,
                                              self.attn_length,
                                              self.c_input,
                                              self.patch_size,
                                              self.embedding_dim)

        self.translayer = TransLayer(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=mlp_dim,
            attn_dropout_rate=dropout_rate,
            ff_dropout_rate=dropout_rate,
        )

        self.context_dropout_rate = context_dropout
        self.context_dropout = nn.Dropout(context_dropout)

        self.context_aggregation = ailoc.transloc.ConvNextBlock(c_in=self.c_input*2,
                                                                c_out=self.c_input,
                                                                kernel_size=3,)
        # self.context_aggregation = nn.Conv2d(in_channels=self.c_input * 2,
        #                                      out_channels=self.c_input,
        #                                      kernel_size=3,
        #                                      padding=1, )

    def forward(self, x):
        input = x
        x, attn_mask = self.embedding_layer(x)
        x = self.translayer(x, attn_mask)
        x = self.embedding_layer.backward(x)
        x = x * self.context_dropout(torch.ones(input.shape[0], input.shape[1], 1, 1, 1).to(x.device)) * (1. - self.context_dropout_rate)
        x = torch.cat([input.reshape([-1, self.c_input, input.shape[-2], input.shape[-1]]),
                       x.reshape([-1, self.c_input, input.shape[-2], input.shape[-1]])], dim=1)
        x = self.context_aggregation(x)
        return x
