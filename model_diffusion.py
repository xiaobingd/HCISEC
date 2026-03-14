import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
#  辅助组件：坐标编码 (解决 Time-Alignment 问题)
# ==========================================
class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Input: [B, C, H, W] -> Output: [B, C+2, H, W]
        通过注入坐标通道，让模型感知绝对的时频位置。
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        # Frequency Axis (H)
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        xx_channel = xx_channel.float() / (x_dim - 1)
        xx_channel = xx_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        # Time Axis (W)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1)
        yy_channel = yy_channel.float() / (y_dim - 1)
        yy_channel = yy_channel * 2 - 1
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        ret = torch.cat([input_tensor, 
                         xx_channel.type_as(input_tensor), 
                         yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + 
                            torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            # 适应 Skip Connection 的通道拼接
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(B, C, H, W)

class BiometricUNet64(nn.Module):
    def __init__(self, input_dim=1, condition_dim=1, time_dim=256):
        super().__init__()
        
        # 1. 坐标通道注入
        self.add_coords = AddCoords(with_r=False)
        # Channels: Input(1) + Condition(1) + Coords(2) = 4
        channels = input_dim + condition_dim + 2 
        
        down_channels = [64, 128]
        out_dim = input_dim 

        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(channels, down_channels[0], 3, padding=1)

        # Downsample
        self.down1 = Block(down_channels[0], down_channels[1], time_dim) # 64 -> 128
        self.down2 = Block(down_channels[1], down_channels[1]*2, time_dim) # 128 -> 256

        # Bottleneck
        self.bottleneck_dim = down_channels[1] * 2 
        self.sa1 = AttentionBlock(self.bottleneck_dim)
        self.conv_bot = nn.Conv2d(self.bottleneck_dim, self.bottleneck_dim, 3, padding=1)
        self.sa2 = AttentionBlock(self.bottleneck_dim)

        # Upsample
        self.up1 = Block(down_channels[1], down_channels[1], time_dim, up=True) 
        self.up2 = Block(down_channels[1], down_channels[0], time_dim, up=True) 

        self.output = nn.Conv2d(down_channels[0] * 2, out_dim, 1)

    def forward(self, x, t, condition):
        t = self.time_mlp(t)
        # Concatenate Input + Condition
        x_in = torch.cat([x, condition], dim=1) 
        # Inject Coordinates
        x_in = self.add_coords(x_in) 

        x1 = self.conv0(x_in)   
        x2 = self.down1(x1, t)  
        x3 = self.down2(x2, t)     

        x3 = self.sa1(x3)
        x3 = self.conv_bot(x3)
        x3 = self.sa2(x3)

        x = self.up1(x3, t)        
        x = torch.cat([x, x2], dim=1) 
        
        x = self.up2(x, t)      
        x = torch.cat([x, x1], dim=1) 

        return self.output(x)