import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),


        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DLKSA(nn.Module):
    def __init__(self, dim, k_size=11, offset_down=2):
        super().__init__()

        if k_size == 11:
            k1, k2, d = 5, 3, 2
        else:
            print("选择k大小")
            k1, k2, d = 3, 5, 2

        self.offset_down = offset_down

        self.conv0h = nn.Conv2d(
            dim, dim, (1, k1),
            padding=(0, (k1 - 1) // 2),
            groups=dim
        )
        self.conv0v = nn.Conv2d(
            dim, dim, (k1, 1),
            padding=((k1 - 1) // 2, 0),
            groups=dim
        )


        offset_channels = 2 * 3*3 # k2
        hidden = max(8, dim // 4)

        self.offset = nn.Sequential(
            nn.Conv2d(dim,hidden , 1),
            nn.GELU(),
            nn.Conv2d(hidden, offset_channels,3,padding=1)
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

        self.deform = DeformConv2d(
            dim, dim,
            kernel_size=3,
            padding=2,
            dilation=d,
            groups=dim,
            bias=False
        )

        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        identity = x

        attn = self.conv0h(x)
        attn = self.conv0v(attn)

        attn_ds = F.avg_pool2d(attn, self.offset_down)
        offset = self.offset(attn_ds)
        offset = F.interpolate(offset, size=attn.shape[-2:], mode='nearest')
        attn = self.deform(attn, offset)

        attn = self.proj(attn)

        return identity * (1.0 + self.scale * attn)

class ChannelHighFreqEnergy(nn.Module):
    def __init__(self, ratio=0.4, eps=1e-6):
        super().__init__()
        self.ratio = ratio
        self.eps = eps
        self.register_buffer('mask_cache', torch.empty(0), persistent=False)
        self.cached_shape = None

    def forward(self, x):
        B, C, H, W = x.shape
        Wf = W // 2 + 1

        if self.mask_cache.numel() == 0 or self.cached_shape != (H, Wf):
            self.cached_shape = (H, Wf)
            yy = torch.arange(H, device=x.device).view(-1, 1)
            xx = torch.arange(Wf, device=x.device).view(1, -1)

            fy = torch.minimum(yy, H - yy)
            fx = xx
            radius = torch.sqrt((fy.float() / H) ** 2 + (fx.float() / W) ** 2)
            mask = (radius >= self.ratio).to(x.dtype)

            self.mask_cache = mask

        X = torch.fft.rfft2(x, norm='ortho')
        power = X.abs() ** 2
        high_freq_power = power * self.mask_cache
        energy = high_freq_power.sum(dim=(-2, -1), keepdim=True)
        energy = torch.log1p(energy)

        mean = energy.mean(dim=1, keepdim=True)
        std = energy.std(dim=1, keepdim=True, unbiased=False) + self.eps
        energy = (energy - mean) / std

        return energy

class SCCM(nn.Module):
    def __init__(self, dim=36,ffn_scale=2.0):
        super().__init__()
        self.dim = dim
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lka =DLKSA(dim,k_size=11)
        self.freq_energy = ChannelHighFreqEnergy(ratio=0.4)

        self.lde = DMlp(dim, ffn_scale)
        self.gelu = nn.GELU()
        self.down_scale = 4 # 8

        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, dim, 1, 1)*0.5)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_pool = F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale))
        x_s = self.lka(x_pool)
        x_v = self.freq_energy(x_pool)
        x_fused = self.gelu(self.linear_1(x_s * self.alpha + x_v * self.beta ))
        x_fused = F.interpolate(x_fused, size=(h, w), mode='nearest')
        x_out = x * x_fused
        y_d = self.lde(y)
        return self.linear_2(x_out+ self.gamma * y_d)

class SGFN(nn.Module):
    def __init__(self, dim, expansion_factor=2.0):
        super(SGFN, self).__init__()
        hidden_dim = int(dim * expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim
        )
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv(x1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class FMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0,block_id=0,n_blocks=8):
        super().__init__()
        self.block_id = block_id

        self.sccm = SCCM(dim)
        self.ffn = SGFN(dim,ffn_scale)
        self.norm1 = LayerNorm(dim,data_format="channels_first")
        self.norm2 = LayerNorm(dim,data_format="channels_first")

    def forward(self, x):
        x = self.sccm(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x


# @ARCH_REGISTRY.register()
class SCMNet(nn.Module):
    def __init__(self, dim=36, n_blocks=8, ffn_scale=2, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.feats = nn.Sequential(*[FMB(dim, ffn_scale,block_id,n_blocks) for block_id in range(n_blocks)]) # FMB
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )
    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x
