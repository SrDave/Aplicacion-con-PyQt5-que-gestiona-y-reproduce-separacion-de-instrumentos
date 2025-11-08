import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.constant_(self.attn.in_proj_bias, 0.)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        nn.init.constant_(self.attn.out_proj.bias, 0.)

    def forward(self, x):
        return self.attn(x, x, x)[0]

class LowLevelAttention(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv_att = nn.Sequential(
            nn.Conv2d(channels, channels//8, kernel_size=(1,15), padding=(0,7)),
            nn.ReLU(),
            nn.Conv2d(channels//8, channels, kernel_size=(1,15), padding=(0,7)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return x * self.conv_att(x)

class DenseHybridAttention(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        # Dense path
        self.dense_conv = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim//2, dim, kernel_size=3, padding=1)
        )
        # Attention path
        self.mha = LayerMultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(dim)
        # Fusion
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # Dense path
        x_dense = self.dense_conv(x)
        
        # Attention path
        B, C, F, T = x.shape
        x_att = x.permute(0, 2, 3, 1).reshape(B*F, T, C)
        x_att = self.norm(x_att)
        x_att = self.mha(x_att)
        x_att = x_att.reshape(B, F, T, C).permute(0, 3, 1, 2)
        
        return x + x_dense + self.gamma * x_att

class DenseSkipConnection(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels + in_channels//2, in_channels//2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels + in_channels, in_channels, kernel_size=1)

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        x1 = F.relu(self.conv1(x))
        x = torch.cat([x, x1], dim=1)
        x2 = F.relu(self.conv2(x))
        x = torch.cat([x, x2], dim=1)
        return self.conv3(x)

class DiffusionRefinement(nn.Module):
    def __init__(self, channels=2):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Bottleneck attention
        self.bottleneck_attn = LayerMultiheadAttention(embed_dim=128, num_heads=8)

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(1,5), padding=(0,2)),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(1,5), padding=(0,2)),
            nn.ReLU()
        )
        self.dec3 = nn.ConvTranspose2d(32, channels, kernel_size=(1,5), padding=(0,2))

        # Final refinement
        self.refine = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.xavier_uniform_(self.refine.weight)
        nn.init.constant_(self.refine.bias, 0.)

    def forward(self, x, noise=None):
        x_noisy = x + 0.1 * (noise if noise is not None else torch.randn_like(x))

        # Encoder
        x1 = self.enc1(x_noisy)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Attention bottleneck
        B, C, F, T = x3.shape
        x_att = self.bottleneck_attn(x3.permute(0, 2, 3, 1).reshape(B*F, T, C))
        x3 = x_att.reshape(B, F, T, C).permute(0, 3, 1, 2)

        # Decoder with skip connections
        x = self.dec1(x3) + x2
        x = self.dec2(x) + x1
        return self.refine(self.dec3(x))

class MusicSeparationModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.enc6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        # Attention modules
        self.attn_x3 = LowLevelAttention(channels=128)
        self.attn_x4 = DenseHybridAttention(dim=256)
        
        # Dense skip connections
        self.dense_skip1 = DenseSkipConnection(1024)
        self.dense_skip2 = DenseSkipConnection(512)
        self.dense_skip3 = DenseSkipConnection(256)
        self.dense_skip4 = DenseSkipConnection(128)
        self.dense_skip5 = DenseSkipConnection(64)

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(512*2, 256, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256*2, 128, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
            nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(128*2, 64, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
            nn.ReLU()
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 32, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
            nn.ReLU()
        )
        self.dec6 = nn.ConvTranspose2d(32*2, 2, kernel_size=(5,1), stride=(2,1), padding=(2,0))

        # Diffusion refinement
        self.diffusion = DiffusionRefinement()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.attn_x3(self.enc3(x2))
        x4 = self.attn_x4(self.enc4(x3))
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)

        # Decoder with dense connections
        x = self.dense_skip1(self.dec1(x6), x5)
        x = self.dense_skip2(self.dec2(x), x4)
        x = self.dense_skip3(self.dec3(x), x3)
        x = self.dense_skip4(self.dec4(x), x2)
        x = self.dense_skip5(self.dec5(x), x1)
        x = self.dec6(x)

        return self.diffusion(x)

# Instanciar modelo mejorado
model = MusicSeparationModel()
#print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
#print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")