import torch.nn as nn;
import torch;
from clone_models.Unet_component import ResnetBlock,SinusoidalPosEmb,Upsample,Downsample,LinearAttention,Residual,Rezero,Block;

class Unet(nn.Module):
    def __init__(self, 
                 dim=64, 
                 out_dim=3, 
                 dim_mults=(1, 2, 2, 4), 
                 cond_dim=32,
                 rrdb_num_block = 8,
                 sr_scale = 4,
                 use_attn = True,
                 res = True,
                 up_input = False):
        
        super().__init__();
        
        self.res = res;
        self.up_input = up_input;
        self.use_attn = use_attn;
        
        dims = [3, *map(lambda m: dim * m, dim_mults)]; #[3,2*dim,4*dim,8*dim]     
        in_out = list(zip(dims[:-1], dims[1:])); #[(3,2*dim),(2*dim,4*dim),(4*dim,8*dim)]
        groups = 0

        self.cond_proj = nn.ConvTranspose2d(cond_dim * ((rrdb_num_block + 1) // 3),
                                            dim, 
                                            sr_scale * 2, 
                                            sr_scale,
                                            sr_scale // 2)
        
        # self.cond_proj = nn.Conv2d(3,dim,kernel_size=sr_scale*2+1,padding=sr_scale,stride=1)

        self.time_pos_emb = SinusoidalPosEmb(dim);
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Mish(inplace=True),
            nn.Linear(dim * 4, dim)
        );

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]; #[8*dim]
        
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        if use_attn:
            self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups),
            nn.Conv2d(dim, out_dim, 1)
        )

        if res and up_input:
            self.up_proj = nn.Sequential(
                nn.ReflectionPad2d(1), 
                nn.Conv2d(3, dim, 3),
            )

    def forward(self, 
                x : torch.Tensor, 
                time : torch.Tensor, 
                cond : list, 
                img_lr_up: torch.Tensor):
        """
        x : [B,C,H,W]
        time : [B,]
        cond : [nb+1,B,3,h,w]
        img_lr_up : [B,C,H,W]
        """
        
        t = self.time_pos_emb(time); #[B,]->[B,D]
        t = self.mlp(t); #[B,D]->[B,D]

        h = []
        # cond_proj will make h->H by sr_scale
        cond = self.cond_proj(torch.cat(list(cond[2::3]), 1)); #[nb+1,B,32,h,w]->[B,3*32,h,w]->[B,96,H,W];

        #cond = self.cond_proj(cond); #[B,3,H,W] -> [B,3,H,W]
        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x, t); 
            x = resnet2(x, t);
            if i == 0:
                x = x + cond;
                if self.res and self.up_input:
                    x = x + self.up_proj(img_lr_up)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t) #[B,8*D,H//8,W//8] + [B,8*D,H//8,W//8]->[B,8*D,H//8,W//8];
        if self.use_attn:
            x = self.mid_attn(x); #[B,8*D,H//8,W//8]
        x = self.mid_block2(x, t) # [B,8*D,H//8,W//8] + [B,8*D,H//8,W//8]->[B,8*D,H//8,W//8]

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)
        # x : [B,8*D,H//8,W//8] -> [B,3,H,W]
        return self.final_conv(x) # [B,out_dim,H,W]