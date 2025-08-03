import torch.nn as nn;
import torch;
import functools;
from clone_utils.utils import make_layer;
import torch.nn.functional as F;

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias);
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias);
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias);
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias);
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias);
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True);

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        # x : [B,64,H,W]
        x1 = self.lrelu(self.conv1(x)); #[B,64,h,w]->[B,32,h,w]
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1))); #[B,32+64,h,w]->[B,32,h,w]
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1))); #[B,2*32+64,32,h,w]->[B,32,h,w]
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1))); #[B,3*32+64,h,w]->[B,32,h,w]
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))# [B,4*32 + 64,h,w]->[B,64,h,w]
        return x5 * 0.2 + x; #[B,64,h,w] + [B,64,h,w]->[B,64,h,w];


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc);
        self.RDB2 = ResidualDenseBlock_5C(nf, gc);
        self.RDB3 = ResidualDenseBlock_5C(nf, gc);

    def forward(self, x):
        # x : [B,nf,h,w]
        out = self.RDB1(x); #[B,nf,h,w]->[B,nf,h,w]
        out = self.RDB2(out); #[B,nf,h,w]->[B,nf,h,w]
        out = self.RDB3(out); #[B,nf,h,w]->[B,nf,h,w]
        return out * 0.2 + x; #[B,nf,h,w] + [B,nf,h,w] -> [B,nf,h,w]
    
    
    
class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=8, gc=32,sr_scale = 4):
        super(RRDBNet, self).__init__();
        
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc);
        self.sr_scale = sr_scale;
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True);
        self.RRDB_trunk = make_layer(RRDB_block_f, nb); 
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True);
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True);
        if sr_scale == 8:
            self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2);

    def forward(self, x : torch.Tensor, get_fea=False):
        # x : [B,3,h,w], low res image
        feas = [];
        x = (x + 1) / 2; 
        fea_first = fea = self.conv_first(x); # [B,32,h,w]
        for l in self.RRDB_trunk:
            fea = l(fea); #[B,32,h,w]
            feas.append(fea)
        trunk = self.trunk_conv(fea); #[B,32,h,w]
        fea = fea_first + trunk #[B,32,h,w]+[B,32,h,w] => [B,32,h,w]
        feas.append(fea);

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='bicubic'))); #[B,32,h,w]->[B,32,h*2,w*2]->[B,32,2h,2w];
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='bicubic'))); #[B,32,2h,2w]->[B,32,4h,4w]
        if self.sr_scale == 8:
            fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='bicubic')))
        
        fea_hr = self.HRconv(fea); #[B,32,4h,4w]->[B,32,4h,4w];
        out = self.conv_last(self.lrelu(fea_hr));#[B,32,4h,4w]->[B,3,4h,4w]
        out = out.clamp(0, 1);
        out = out * 2 - 1
        if get_fea:
            return out, feas; # [B,3,4h,4w], [nb+1,B,32,4h,4w]-> [9,B,32,4h,4w] -> [9,B,32,H,W]
        else:
            return out;