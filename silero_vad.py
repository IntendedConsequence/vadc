import torch
import math
import torch.nn.functional as F

def pad_reflect(x, pad: int):
    return F.pad(x, [pad, pad], 'reflect')

class STFT(torch.nn.Module):
    def __init__(self, n_fft=256, is_v4=False) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.stride = n_fft // 4
        self.to_pad = int((n_fft - self.stride) / 2) if is_v4 else n_fft // 2
        self.hann = torch.hann_window(self.n_fft)
        self.forward_basis_buffer = torch.nn.Parameter(torch.zeros((n_fft+2, 1, n_fft)), requires_grad=False)

    def forward(self, input: torch.Tensor):
        return torch.stft(pad_reflect(input, self.to_pad), n_fft=self.n_fft, center=False, hop_length=self.stride, win_length=self.n_fft, window=self.hann, return_complex=True).abs()

class AdaptiveAudioNormalization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_pad = 3
        self.filter_ = torch.nn.Parameter(torch.zeros((1, 1, 7)), requires_grad=False)

    def forward(self, spect: torch.Tensor) -> torch.Tensor:
        spect_e = torch.log1p(spect * 1048576)
        if len(spect_e.shape) == 2:
            spect_e = spect_e[None, :, :]
        mean = spect_e.mean(dim=1, keepdim=True)
        mean_padded = pad_reflect(mean, self.to_pad)
        mean_padded_convolved = torch.conv1d(mean_padded, self.filter_)
        mean_mean = mean_padded_convolved.mean(dim=-1, keepdim=True)
        normalized = spect_e - mean_mean
        return normalized

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int = 129, out_channels: int = 16, has_out_proj: bool = True):
        super().__init__()
        self.dw_conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, padding=2, groups=in_channels),
            torch.nn.Identity(),
            torch.nn.ReLU()
        )
        self.pw_conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            torch.nn.Identity()
        )
        if has_out_proj:
            self.proj = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.proj = torch.nn.Identity()
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pw_conv(self.dw_conv(x))
        x += self.proj(residual)
        x = self.activation(x)
        return x

# v3 only
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, qkv_in_features: int, qkv_out_features: int, n_heads: int = 2):
        super().__init__()
        self.head_dim = qkv_in_features / n_heads
        self.scale = math.sqrt(self.head_dim)
        self.n_heads = n_heads

        self.QKV = torch.nn.Linear(in_features=qkv_in_features, out_features=qkv_out_features)
        self.out_proj = torch.nn.Linear(in_features=qkv_in_features, out_features=qkv_in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        head_dim = dim // self.n_heads

        q, k, v = self.QKV(x).chunk(3, dim=-1)

        # split heads - process them independently, just Like different elements in the batch
        # (bs, seq, hid) -> (seq, bs * head, hid / head) -> (bs * head, seq, hid / head)
        k = k.transpose(0, 1).contiguous().view(seq, bsz * self.n_heads, head_dim).transpose(0, 1)
        q = q.transpose(0, 1).contiguous().view(seq, bsz * self.n_heads, head_dim).transpose(0, 1)
        v = v.transpose(0, 1).contiguous().view(seq, bsz * self.n_heads, head_dim).transpose(0, 1)

        # (bs * head, seq, hid/head) @ (bs / head, hid / head, seq)
        alpha = F.softmax(k @ q.transpose(1, 2) / self.scale, dim=-1)

        # (bs * head, seq, seq) @ (bs * head, seq, hid / head)
        attn = alpha @ v

        # (bs * head, seg, hid / head) -> (seq, bs * head, hid / head) ->  (seq, bs, hid) ->  (bs, seq, hid)
        attn = attn.transpose(0, 1).contiguous().view(seq, bsz, dim).transpose(0, 1)
        attn = self.out_proj(attn)

        return attn

# v3 only
class TransformerLayer(torch.nn.Module):
    def __init__(self, shape: int, att_qkv_in: int, att_qkv_out: int):
        super().__init__()

        self.attention = MultiHeadAttention(qkv_in_features=att_qkv_in, qkv_out_features=att_qkv_out)
        self.activation = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.norm1 = torch.nn.LayerNorm(normalized_shape=shape)
        self.norm2 = torch.nn.LayerNorm(normalized_shape=shape)
        self.linear1 = torch.nn.Linear(in_features=shape, out_features=shape)
        self.linear2 = torch.nn.Linear(in_features=shape, out_features=shape)

    def forward(self, x) -> torch.Tensor:
        # (batch * dims * sequence) => (batch * sequence * dims)
        x = x.permute(0, 2, 1).contiguous()

        attn = self.attention(x)
        x = x + self.dropout1(attn)
        x = self.norm1(x)

        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        # (batch * sequence * dims) => (batch * dims * sequence)
        x = x.permute(0, 2, 1).contiguous()
        return x

def encoder(is_v4 = False, sr = 16000):
    enc = []
    if not is_v4:
        enc.append(TransformerLayer(shape=16, att_qkv_in=16, att_qkv_out=48))
    enc.append(torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=2))
    enc.append(torch.nn.BatchNorm1d(16))
    enc.append(torch.nn.ReLU())

    enc.append(torch.nn.Sequential(ConvBlock(in_channels=16, out_channels=32)))
    if not is_v4:
        enc.append(TransformerLayer(shape=32, att_qkv_in=32, att_qkv_out=96))
    enc.append(torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=2))
    enc.append(torch.nn.BatchNorm1d(32))
    enc.append(torch.nn.ReLU())

    enc.append(torch.nn.Sequential(ConvBlock(in_channels=32, out_channels=32, has_out_proj=False)))
    if not is_v4:
        enc.append(TransformerLayer(shape=32, att_qkv_in=32, att_qkv_out=96))
    if is_v4 and sr == 16000:
        enc.append( torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=2))
    else:
        enc.append( torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1))
    enc.append(torch.nn.BatchNorm1d(num_features=32))
    enc.append(torch.nn.ReLU())

    enc.append(torch.nn.Sequential(ConvBlock(in_channels=32, out_channels=64)))
    if not is_v4:
        enc.append(TransformerLayer(shape=64, att_qkv_in=64, att_qkv_out=192))
    enc.append(torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1))
    enc.append(torch.nn.BatchNorm1d(num_features=64))
    enc.append(torch.nn.ReLU())

    return torch.nn.Sequential(*enc)

class Silero_V4(torch.nn.Module):
    def __init__(self, sr=16000):
        super().__init__()

        self.feature_extractor = STFT(256, is_v4=True)
        self.adaptive_normalization = AdaptiveAudioNormalization()
        self.first_layer = torch.nn.Sequential(ConvBlock(258))
        self.encoder = encoder(is_v4=True, sr=sr)

        self.decoder = torch.nn.ModuleDict({
            "rnn": torch.nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1),
            "decoder": torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1),
                torch.nn.Sigmoid()
            )
        })

    def forward(self, input, h, c):
        spect = self.feature_extractor(input)
        normalized = self.adaptive_normalization(spect)
        first_layer_out = self.first_layer(torch.cat([spect, normalized], 1))
        encoder_out = self.encoder(first_layer_out)

        encoder_out_t = torch.permute(encoder_out, [0, 2, 1])

        lstm_out, (hn, cn) = self.lstm_minibatched(encoder_out_t, h, c)

        lstm_out_t = torch.permute(lstm_out, [0, 2, 1])
        decoder_out = self.decoder.decoder(lstm_out_t)

        out = torch.unsqueeze(torch.mean(torch.squeeze(decoder_out, 1), [1]), 1)
        return out, hn, cn

    def lstm_unbatched(self, encoder_out_t: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        return self.decoder.rnn(encoder_out_t, (h, c))

    def lstm_minibatched_(self, encoder_out_t: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        return lstm_minibatched(self.decoder.rnn, encoder_out_t, h, c)

    def lstm_minibatched(self, encoder_out_t: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        batch, seq, feat = encoder_out_t.shape
        r, (h, c) = self.decoder.rnn(encoder_out_t.reshape(1, -1, feat), (h, c))
        r = r.reshape(batch, seq, -1)

        return r, (h, c)

def lstm_minibatched(lstm: torch.nn.LSTM, input: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
    batch, seq, feat = input.shape
    r, (h, c) = lstm(input.reshape(1, -1, feat), (h, c))
    r = r.reshape(batch, seq, -1)

    return r, (h, c)

class Silero_V3(torch.nn.Module):
    def __init__(self, sr=16000):
        super().__init__()

        self.feature_extractor = STFT(256 if sr==16000 else 128, is_v4=False)
        self.adaptive_normalization = AdaptiveAudioNormalization()
        self.first_layer = torch.nn.Sequential(ConvBlock(129 if sr==16000 else 65))
        self.encoder = encoder(is_v4=False, sr=sr)

        self.lstm = torch.nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        self.decoder = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=64, out_channels=2, kernel_size=1),
            torch.nn.AdaptiveAvgPool1d(output_size=1),
            torch.nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        spect = self.feature_extractor(input)
        normalized = self.adaptive_normalization(spect)
        first_layer_out = self.first_layer(normalized)
        encoder_out = self.encoder(first_layer_out)

        encoder_out_t = torch.permute(encoder_out, [0, 2, 1])
        lstm_out, (h0, c0), = self.lstm_minibatched(encoder_out_t, h, c)
        lstm_out_t = torch.permute(lstm_out, [0, 2, 1])
        out = self.decoder(lstm_out_t)
        return out, h0, c0

    def lstm_unbatched(self, encoder_out_t: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        return self.lstm(encoder_out_t, (h, c))

    def lstm_minibatched_(self, encoder_out_t: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        return lstm_minibatched(self.lstm, encoder_out_t, h, c)

    def lstm_minibatched(self, encoder_out_t: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        batch, seq, feat = encoder_out_t.shape
        r, (h, c) = self.lstm(encoder_out_t.reshape(1, -1, feat), (h, c))
        r = r.reshape(batch, seq, -1)

        return r, (h, c)
