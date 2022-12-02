import torch
import torch.nn as nn
from timm.models.layers import to_2tuple,DropPath

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  
        #print(x.shape)
        qkv = self.qkv(x)
        qkv=qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv=qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        
        print(q.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class MY_1(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.window_partition=window_partition
                
        self.conv0_1 = double_conv(3,16)
        self.conv0_2 = double_conv(3,16)
        self.conv0_3 = double_conv(3,16)
        self.conv0_4 = double_conv(3,16)  
        
        self.conv1_1 = double_conv(64,32)
        self.conv1_2 = double_conv(64,32)
        self.conv1_3 = double_conv(64,32)
        self.conv1_4 = double_conv(64,32)
        
        self.conv2_1 = double_conv(128,64)
        self.conv2_2 = double_conv(128,64)
        self.conv2_3 = double_conv(128,64)
        self.conv2_4 = double_conv(128,64)
        
        self.conv3_1 = double_conv(256,128)
        self.conv3_2 = double_conv(256,128)
        self.conv3_3 = double_conv(256,128)
        self.conv3_4 = double_conv(256,128)
          
        self.conv4_1 = double_conv(512,512)
        self.conv4_2 = double_conv(512,512)
        self.conv4_3 = double_conv(512,512)
        self.conv4_4 = double_conv(512,512)
        
        self.activation = torch.nn.Sigmoid()
        
        self.conv_B0 = conv_bank(128,128)
        self.conv_B1 = conv_bank(256,128)
        self.conv_B2 = conv_bank(96,64)
        self.conv_B3 = conv_bank(48,32)
        self.conv_B4 = conv_bank(24,16)
        
        self.conv_ =nn.Conv2d(16,1,1)
        
        dpr = [x.item() for x in torch.linspace(0,0, 4)]  # stochastic depth decay rule
        
        
        self.blocks1 = nn.ModuleList([
            Block(
                dim=14*14, num_heads=4, mlp_ratio=1, qkv_bias=False, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(4)])
