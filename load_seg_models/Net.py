import torch
import torch.nn as nn
import torch.nn.functional as F

def crop(upsampled, bypass):
    h1, w1 = upsampled.shape[2], upsampled.shape[3]
    h2, w2 = bypass.shape[2], bypass.shape[3]
    
    deltah = h2 - h1
    deltaw = w2 - w1
    
    pad_top = deltah // 2
    pad_bottom = deltah - pad_top
    pad_left = deltaw // 2
    pad_right = deltaw - pad_left
    
    upsampled_padded = F.pad(upsampled, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
    return upsampled_padded

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)

class CBAM(nn.Module):
    """卷积块注意力模块"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1, use_attention=False):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_attention:
            out = self.attention(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class SegNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SegNet, self).__init__()
        # 编码器 - 在特征层次变化处加注意力
        self.encoder1 = nn.Sequential(
            ResidualBlock(3, 64, use_attention=True),    # 输入层：关注基础特征
            ResidualBlock(64, 64, use_attention=False)
        )
        
        self.encoder2 = nn.Sequential(
            ResidualBlock(64, 128, use_attention=True),  # 特征维度变化：关注重要通道
            ResidualBlock(128, 128, use_attention=False)
        )
        
        self.encoder3 = nn.Sequential(
            ResidualBlock(128, 256, use_attention=False),
            ResidualBlock(256, 256, use_attention=False),
            ResidualBlock(256, 256, use_attention=True)   # 深层特征：关注语义信息
        )
        
        self.encoder4 = nn.Sequential(
            ResidualBlock(256, 512, use_attention=False),
            ResidualBlock(512, 512, use_attention=False),
            ResidualBlock(512, 512, use_attention=False)
        )
        
        self.encoder5 = nn.Sequential(
            ResidualBlock(512, 512, use_attention=True),  # 最深层：全局注意力
            ResidualBlock(512, 512, use_attention=False),
            ResidualBlock(512, 512, use_attention=False)
        )
        
        # 解码器 - 在特征融合处加注意力
        self.decoder1 = nn.Sequential(
            ResidualBlock(512, 512, use_attention=True),  # 解码开始：关注特征重建
            ResidualBlock(512, 512, use_attention=False),
            ResidualBlock(512, 512, use_attention=False)
        )
        
        self.decoder2 = nn.Sequential(
            ResidualBlock(512, 512, use_attention=False),
            ResidualBlock(512, 512, use_attention=False),
            ResidualBlock(512, 256, use_attention=False)
        )
        
        self.decoder3 = nn.Sequential(
            ResidualBlock(256, 256, use_attention=False),
            ResidualBlock(256, 256, use_attention=False),
            ResidualBlock(256, 128, use_attention=False)
        )
        
        self.decoder4 = nn.Sequential(
            ResidualBlock(128, 128, use_attention=True),  # 接近输出：关注边界细节
            ResidualBlock(128, 64, use_attention=False)
        )
        
        self.decoder5 = nn.Sequential(
            ResidualBlock(64, 64, use_attention=False),
            ResidualBlock(64, 64, use_attention=False),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

        # 池化和反池化
        self.max_pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(2, 2)
        
        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 编码器
        x1 = self.encoder1(x)
        x, pool_indices1 = self.max_pool(x1)
        
        x2 = self.encoder2(x)
        x, pool_indices2 = self.max_pool(x2)
        
        x3 = self.encoder3(x)
        x, pool_indices3 = self.max_pool(x3)
        
        x4 = self.encoder4(x)
        x, pool_indices4 = self.max_pool(x4)
        
        x5 = self.encoder5(x)
        x, pool_indices5 = self.max_pool(x5)
        
        # 解码器
        x = self.max_unpool(x, pool_indices5)
        x = crop(x, x5)
        x = self.decoder1(x)
        
        x = self.max_unpool(x, pool_indices4)
        x = crop(x, x4)
        x = self.decoder2(x)
        
        x = self.max_unpool(x, pool_indices3)
        x = crop(x, x3)
        x = self.decoder3(x)
        
        x = self.max_unpool(x, pool_indices2)
        x = crop(x, x2)
        x = self.decoder4(x)
        
        x = self.max_unpool(x, pool_indices1)
        x = crop(x, x1)
        x = self.decoder5(x)
        
        return x

if __name__ == '__main__':
    model = SegNet(num_classes=2)
    print("优化版SegNet创建成功！")
    print("注意力机制位置：")
    print("  - 编码器: 输入层、特征维度变化处、深层语义层、最深层")
    print("  - 解码器: 特征重建层、边界细节层")
    img = torch.randn(1, 3, 128, 128)
    output = model(img)
    print(f"输入: 1x3x128x128, 输出: {output.shape}")