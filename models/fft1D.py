import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import torch
import torch.nn as nn

from ..attention import FullAttention, ProbAttention, AttentionLayer


# ---------------------- 动态位置编码类 ----------------------
class DynamicPosition(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        # 可学习的位置参数
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))
        #self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)  # 增加尺度控制
        #self.pos_embed = nn.Parameter(self._get_sinusoidal_encoding(max_len, d_model))
        # 动态衰减系数
        self.decay = nn.Parameter(torch.linspace(1, 0, d_model))
          
    def forward(self, x):
        B, L, D = x.shape
        # 截取对应长度位置编码
        pos = self.pos_embed[:, :L] * self.decay.view(1,1,-1)
        return pos  # (1, L, D)

# ---------------------- 修改数据嵌入层 ----------------------
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embed = nn.Linear(c_in, d_model)
        self.position = DynamicPosition(d_model)  # 替换原位置编码
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.value_embed(x) + self.position(x)  # 动态位置注入
        return self.dropout(x)


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        #self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        #self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.ffn1 = nn.Linear(d_model, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        #y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        #y = self.dropout(self.conv2(y).transpose(-1,1))
        y = self.dropout(self.activation(self.ffn1(y)))
        y = self.ffn2(y)

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


# ---------------------- 改进的复数投影 ----------------------
class ComplexProjection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.real_proj = nn.Linear(d_model, d_model)
        self.imag_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x_fft):
        # x_fft: complex tensor [batch_size, seq_len//2+1, d_model]
        real = self.real_proj(x_fft.real)
        imag = self.imag_proj(x_fft.imag)
        return real + imag


# ---------------------- 修正后的傅里叶混合层 ----------------------
class FourierMixer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.complex_proj = ComplexProjection(d_model)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x_fft = torch.fft.fft(x, dim=1)  # 沿时间维（seq_len）做FFT
        return self.complex_proj(x_fft)  # 保留复数信息并投影



# ---------------------- 修改后的编码器层 ----------------------
class FNetEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.fourier_mixer = FourierMixer(d_model)
        #self.fourier_mixer = DualPathMixer(d_model)


        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        x = x + self.fourier_mixer(self.norm1(x))  # 残差连接 + 频域建模
        x = x + self.ffn(self.norm2(x))            # 残差连接 + 前馈
        return x, None  # 无注意力输出

# ---------------------- 修改后的编码器 ----------------------
class FNetEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x, _ = layer(x, attn_mask=attn_mask)  # 忽略注意力掩码
        if self.norm is not None:
            x = self.norm(x)
        return x, []  # 返回空注意力列表


# ---------------------- 修改后的Transformer类 ----------------------
class fft1D(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='full', activation='gelu',
                 output_attention=False, distil=False, mix=False,
                 device=torch.device('cuda:0')):
        super(fft1D, self).__init__()
        self.pred_len = out_len
        self.output_attention = False  # 强制关闭注意力输出
        
        # 数据嵌入层保持不变
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        # ---------------------- 修改编码器部分 ----------------------
        self.encoder = FNetEncoder(
            [
                FNetEncoderLayer(
                    d_model=d_model,
                    d_ff=d_ff or 4*d_model,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # ---------------------- 解码器保持不变 ----------------------
        Attn = FullAttention if attn=='full' else ProbAttention
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # 编码流程（使用傅里叶混合）
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out)  # 忽略注意力权重
        
        # 解码流程（保持原注意力机制）
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]