import torch;
import torch.nn as nn;
import math;
from einops import rearrange;

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
        )

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 2),
        )

    def forward(self, x):
        return self.conv(x)

# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim # (time dimension)
        self.theta = theta 

    def forward(self, x : torch.Tensor):
        # x : [B,], time 
        device = x.device
        half_dim = self.dim // 2; 
        emb = math.log(self.theta) / (half_dim - 1); #[Theta,]
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb); #[D//2]
        emb = x[:, None] * emb[None, :]; #[B,1] * [1,D//2]-> [B,D//2]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1); #[B,D//2 + D//2]->[B,D]
        return emb; #[B,D]

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        if groups == 0:
            self.block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim_out, 3),
                nn.Mish()
            );
        else:
            self.block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim_out, 3),
                nn.GroupNorm(groups, dim_out),
                nn.Mish()
            );

    def forward(self, x):
        return self.block(x);

# Block that merge with time component
class ResnetBlock(nn.Module):
    
    # All subsequent parameters (after the asterisk) must be specified as keyword arguments, rather than as positional arguments.
    def __init__(self, dim, dim_out, *, time_emb_dim=0, groups=8):
        super().__init__()
        if time_emb_dim > 0:
            self.mlp = nn.Sequential(
                nn.Mish(),
                nn.Linear(time_emb_dim, dim_out)
            )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity() # residual

    def forward(self, x: torch.Tensor, 
                time_emb=None, 
                cond=None):
        """
        x : [B,dim_out,H,W]
        time_emb : [B,dim_out]
        cond : [B,dim_out,H,W]
        """
        # print(x.shape)
        h = self.block1(x); #[B,dim_out,H,W];
        if time_emb is not None:
            h += self.mlp(time_emb)[:, :, None, None] #[B,dim_out,H,W]
        if cond is not None:
            h += cond; #[B,dim_out,H,W]
        h = self.block2(h)
        return h + self.res_conv(x); #[B,dim_out,H,W]
    
    
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out);
    

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g;
    
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
    