import torch
from torch import nn
from einops import rearrange

import ailoc.transloc


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

    def forward(self, x):
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
            input_dropout_rate=0.0,
            attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        # print("trans depth:", depth)
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim=dim,
                            dropout_rate=input_dropout_rate,
                            fn=SelfAttention(dim=dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(
                            dim=dim,
                            fn=FeedForward(dim=dim, hidden_dim=mlp_dim, dropout_rate=input_dropout_rate)
                        )
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers, return_intermediate=False)

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer block, the images features are split into patches, however, the MSA is applied
    to the feature patches along the feature channels, not the spatial dimensions.
    """

    def __init__(
            self,
            seq_length=48,
            hw_input=64,  # CNN feature size
            # channel_in=144,
            # embedding_channels=16,
            embedding_dim=16*16,  # feature patch size
            num_layers=2,
            num_heads=8,
            mlp_dim=16*16*4,  # embedding_dim * 4
            input_dropout_rate=0,
            attn_dropout_rate=0,
    ):
        super(TransformerBlock, self).__init__()
        self.seq_length = seq_length
        self.hw_input = hw_input
        # self.channel_in = channel_in
        # self.embedding_channels = embedding_channels
        self.embedding_dim = embedding_dim
        self.patch_size = int((embedding_dim)**0.5)

        # self.embedding_conv_in = ailoc.transloc.Conv2d_ELU(channel_in, embedding_channels,1,1,0)

        self.position_encoding = PositionalEncoding(embedding_dim, seq_length)
        # self.pre_dropout = nn.Dropout(p=input_dropout_rate)
        self.translayer = TransLayer(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=mlp_dim,
            input_dropout_rate=input_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        # self.pre_head_ln = nn.LayerNorm(embedding_dim)
        # self.embedding_conv_out = ailoc.transloc.Conv2d_ELU(embedding_channels, channel_in,1,1,0)

    def forward(self, x):
        # x = self.embedding_conv_in(x)
        x = self.split_patches(x)
        batch, channel, patch_num, height, width = x.size()
        # x = rearrange(x, 'b c p h w -> (b p) c (h w)')  # for channel attention
        x = rearrange(x, 'b c p h w -> (b c) p (h w)')  # for spatial attention
        # x = rearrange(x, 'b c p h w -> b p (c h w)')  # for spatial attention, compressed channel as patch
        x = self.position_encoding(x)
        # x = self.pre_dropout(x)
        x = self.translayer(x)
        # x = self.pre_head_ln(x)
        # x = rearrange(x, '(b p) c (h w) -> b c p h w', b=batch, h=height,)  # for channel attention
        x = rearrange(x, '(b c) p (h w) -> b c p h w', b=batch, h=height, )  # for spatial attention
        # x = rearrange(x, 'b p (c h w) -> b c p h w', c=self.embedding_channels, h=height, )  # for spatial attention, compressed channel as patch
        x = self.merge_patches(x)
        # x = self.embedding_conv_out(x)
        return x

    def split_patches(self, x):
        x = rearrange(x, 'b c (h p1) (w p2) -> b c (h w) p1 p2', p1=self.patch_size, p2=self.patch_size)
        return x

    def merge_patches(self, x):
        x = rearrange(x, 'b c (h w) p1 p2 -> b c (h p1) (w p2)', h=int(self.hw_input/self.patch_size),)
        return x

