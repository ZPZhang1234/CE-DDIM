
import math
from typing import Tuple
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        
        q = self.proj_q(h).contiguous()
        k = self.proj_k(h).contiguous()
        v = self.proj_v(h).contiguous()
        
        q = q.view(B, H * W, C)
        k = k.view(B, C, H * W)
        v = v.view(B, H * W, C)

        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        w = F.softmax(w, dim=-1)

        h = torch.bmm(w, v)
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        
        return x + h

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

class UNet_CE_DDIM(nn.Module):

    def __init__(self,
                 T: int,
                 input_channel: int,
                 output_channel: int,
                 ch: int,
                 ch_mult: Tuple[int, ...],
                 attn: Tuple[int, ...],
                 num_res_blocks: int,
                 dropout: float):
        super().__init__()
        tdim = ch * 4                          
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.head = nn.Conv2d(input_channel, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs, now_ch = [ch], ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(now_ch, out_ch, tdim, dropout, attn=(i in attn))
                )
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:             
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(chs.pop() + now_ch, out_ch, tdim, dropout, attn=(i in attn))
                )
                now_ch = out_ch
            if i != 0:                                
                self.upblocks.append(UpSample(now_ch))

        assert len(chs) == 0, "Residual-stack bookkeeping error"
        self.norm_act = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
        )
        self.eps_head    = nn.Conv2d(now_ch, output_channel, 3, 1, 1)   # ε̂
        self.logvar_head = nn.Conv2d(now_ch, output_channel, 3, 1, 1)   # log σ̂²_ε
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)

        init.xavier_uniform_(self.eps_head.weight,    gain=1e-5)
        init.zeros_(self.eps_head.bias)
        init.xavier_uniform_(self.logvar_head.weight, gain=1e-5)
        init.zeros_(self.logvar_head.bias)


    def forward(self, x: torch.Tensor, t: torch.Tensor):

        temb = self.time_embedding(t)


        h  = self.head(x)
        hs = [h]
        for layer in self.downblocks:   
            h = layer(h, temb)
            hs.append(h)
        for layer in self.middleblocks:
            h = layer(h, temb)
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h          = self.norm_act(h)
        eps_hat    = self.eps_head(h)
        logvar_hat = self.logvar_head(h)
        return eps_hat, logvar_hat

class UNet_Con_DDIM(nn.Module):
    def __init__(self, T, input_channel, output_channel, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(input_channel, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, output_channel, 3, stride=1, padding=1) 
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        temb = self.time_embedding(t)
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        for layer in self.middleblocks:
            h = layer(h, temb)
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h

class UNet_baseline(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, ch_mult=(1, 2, 4, 8), num_res_blocks=2, dropout=0.0):
        super().__init__()

        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1) 
        self.downblocks = nn.ModuleList()
        chs = [base_channels]
        now_ch = base_channels

        for i, mult in enumerate(ch_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(now_ch, out_ch, dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.Sequential(
            ResBlock(now_ch, now_ch, dropout),
            ResBlock(now_ch, now_ch, dropout)
        )

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.upblocks.append(ResBlock(now_ch + chs.pop(), out_ch, dropout))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        self.final_upsample = UpSample(base_channels)

        self.tail = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            Swish(),
            nn.Conv2d(base_channels, out_channels, 3, stride=1, padding=1),
            nn.Sigmoid()  
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-2].weight, gain=1e-5) 
        init.zeros_(self.tail[-2].bias)

    def forward(self, x):
        h = self.head(x)
        hs = [h]
        
        for layer in self.downblocks:
            h = layer(h)
            hs.append(h)
        
        h = self.middleblocks(h)
        
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                skip_connection = hs.pop()
                if h.size(2) != skip_connection.size(2) or h.size(3) != skip_connection.size(3):
                    h = F.interpolate(h, size=(skip_connection.size(2), skip_connection.size(3)))
                h = torch.cat([h, skip_connection], dim=1)

            h = layer(h)

        h = self.final_upsample(h)

        h = self.tail(h)
        return h