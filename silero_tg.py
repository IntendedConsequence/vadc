import tinygrad as tg
from tinygrad import nn, Tensor
from tinygrad.helpers import Timing
from tinygrad.jit import TinyJit

from tinygrad import Device
# Device.DEFAULT = "CPU"


import numpy as np

from silero_vad_v3 import chunks_grouped

def Identity(x):
    return x

def load_state_dict_prefix(model, state_dict, prefix=''):
    if hasattr(model, 'load_state_dict'):
        model.load_state_dict(state_dict, prefix)
    else:
        t = {k.replace(prefix, ''): v for k, v in state_dict.items() if k.startswith(prefix)}
        # print(t)
        nn.state.load_state_dict(model, t)

class ConvBlock:
    def __init__(self, in_channels: int = 129, out_channels_pw_proj: int = 16, has_out_proj: bool = True) -> None:
        self.dw_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.pw_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels_pw_proj, kernel_size=1)

        self.has_out_proj = has_out_proj
        if has_out_proj:
            self.proj = nn.Conv1d(in_channels=in_channels, out_channels=out_channels_pw_proj, kernel_size=1)
        else:
            self.proj = lambda x: x

    def __call__(self, x):
        return self.forward(x)

    def load_state_dict(self, state_dict, prefix=''):
        t = {k.replace(prefix, '').replace(".0", ''): v for k, v in state_dict.items() if k.startswith(prefix)}
        nn.state.load_state_dict(self, t)

    def forward(self, x):
        dw_conv_result = self.dw_conv(x).relu()
        proj_result = self.proj(x)
        pw_conv_result = self.pw_conv(dw_conv_result) + proj_result

        return pw_conv_result.relu()


class MultiHeadAttention:
    scale : float
    n_heads : int
    has_out_proj : bool

    def __init__(self, qkv_in_features: int, qkv_out_features: int, scale: float = 2 * np.sqrt(2), n_heads: int = 2):
        self.scale = scale
        self.n_heads = n_heads

        self.QKV = nn.Linear(in_features=qkv_in_features, out_features=qkv_out_features)
        self.has_out_proj = True

        if self.has_out_proj:
            self.out_proj = nn.Linear(in_features=qkv_in_features, out_features=qkv_in_features)
        else:
            self.out_proj = Identity

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        # bsz, seq, dim, = torch.size(x)
        bsz, seq, dim = x.shape
        n_heads = self.n_heads
        head_dim = dim // self.n_heads
        # head_dim = torch.floordiv(dim, n_heads)
        # QKV = self.QKV
        # _2 = torch.chunk((QKV).forward(x, ), 3, -1)
        # q, k, v, = _2
        q, k, v = self.QKV(x).chunk(3, dim=-1)

        # split heads - process them independently, just Like different elements in the batch
        # (bs, seq, hid) -> (seq, bs * head, hid / head) -> (bs * head, seq, hid / head)

        k = k.transpose(0, 1).contiguous().reshape(seq, bsz * self.n_heads, head_dim).transpose(0, 1)

        # _3 = torch.contiguous(torch.transpose(k, 0, 1))
        # n_heads0 = self.n_heads
        # _4 = [seq, torch.mul(bsz, n_heads0), head_dim]
        # k0 = torch.transpose(torch.view(_3, _4), 0, 1)


        q = q.transpose(0, 1).contiguous().reshape(seq, bsz * self.n_heads, head_dim).transpose(0, 1)

        # _5 = torch.contiguous(torch.transpose(q, 0, 1))
        # n_heads1 = self.n_heads
        # _6 = [seq, torch.mul(bsz, n_heads1), head_dim]
        # q0 = torch.transpose(torch.view(_5, _6), 0, 1)

        v = v.transpose(0, 1).contiguous().reshape(seq, bsz * self.n_heads, head_dim).transpose(0, 1)

        # _7 = torch.contiguous(torch.transpose(v, 0, 1))
        # n_heads2 = self.n_heads
        # _8 = [seq, torch.mul(bsz, n_heads2), head_dim]
        # v0 = torch.transpose(torch.view(_7, _8), 0, 1)

        value = k @ q.transpose(1, 2) / self.scale
        alpha = value.softmax(axis=-1)  # (bs * head, seq, hid/head) @ (bs / head, hid / head, seq)
        # _9 = torch.matmul(k, torch.transpose(q, 1, 2))
        # scale = self.scale
        # alpha = _1(torch.div(_9, scale), -1, 3, None, )

        attn = alpha @ v  # (bs * head, seq, seq) @ (bs * head, seq, hid / head)
        # attn = torch.matmul(alpha, v)

        # (bs * head, seg, hid / head) -> (seq, bs * head, hid / head) ->  (seq, bs, hid) ->  (bs, seq, hid)
        attn = attn.transpose(0, 1).contiguous().reshape(seq, bsz, dim).transpose(0, 1)
        # _10 = torch.contiguous(torch.transpose(attn, 0, 1))
        # attn0 = torch.transpose(torch.view(_10, [seq, bsz, dim]), 0, 1)

        attn = self.out_proj(attn)

        return attn

class TransformerLayer:
    def __init__(self, shape: int, att_qkv_in: int, att_qkv_out: int, scale: float = 2 * np.sqrt(2)):
        self.attention = MultiHeadAttention(qkv_in_features=att_qkv_in, qkv_out_features=att_qkv_out, scale=scale)
        # self.activation = torch.nn.ReLU()
        # self.dropout1 = torch.nn.Dropout(0.1)
        # self.dropout = torch.nn.Dropout(0.1)
        # self.dropout2 = torch.nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(normalized_shape=shape)
        self.norm2 = nn.LayerNorm(normalized_shape=shape)
        self.linear1 = nn.Linear(in_features=shape, out_features=shape)
        self.linear2 = nn.Linear(in_features=shape, out_features=shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        # (batch * dims * sequence) => (batch * sequence * dims)
        # if self.reshape_inputs:
        #     x = x.permute(0, 2, 1).contiguous()
        x = x.permute(0, 2, 1).contiguous()

        attn = self.attention(x)
        x = x + attn.dropout(0.1) #dropout1
        x = self.norm1(x)

        x2 = self.linear2(self.linear1(x).relu().dropout(0.1)) #dropout
        x = x + x2.dropout(0.1) #dropout2
        x = self.norm2(x)

        # (batch * sequence * dims) => (batch * dims * sequence)
        # if self.reshape_inputs:
        #     x = x.permute(0, 2, 1).contiguous()
        x = x.permute(0, 2, 1).contiguous()
        return x

# BatchNorm1d = nn.BatchNorm2d
class BatchNorm1d(nn.BatchNorm2d):
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x.unsqueeze(-1)).squeeze(-1)


class Encoder:
    def __init__(self):
        # 0
        transformer = TransformerLayer(shape=16, att_qkv_in=16, att_qkv_out=48, scale=2 * np.sqrt(2))
        self.transformer = transformer

        # 1 full: in_channels=16, out_channels=16, kernel_size=1, stride=2, padding=0, dilation=1, groups=1, padding_mode='zeros'
        conv1d_1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=2)
        self.conv1d_1 = conv1d_1

        # 2 full: num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        batch_norm1d_1 = BatchNorm1d(16)
        self.batch_norm1d_1 = batch_norm1d_1

        # 3
        # relu_1 = torch.nn.ReLU()
        # self.relu_1 = relu_1

        # 4.0, ConvBlock
        conv_block_1 = ConvBlock(in_channels=16, out_channels_pw_proj=32)
        self.conv_block_1 = conv_block_1

        # 5 TransformerLayer
        # att 96
        transformer_layer_1 = TransformerLayer(shape=32, att_qkv_in=32, att_qkv_out=96, scale=4.0)
        self.transformer_layer_1 = transformer_layer_1
        # 5.attention MultiHeadAttention(in_features=32, scale=4.0)
        # 5.norm1 torch.nn.LayerNorm(normalized_shape=32)
        # 5.norm2 torch.nn.LayerNorm(normalized_shape=32)
        # 5.linear1 torch.nn.Linear(in_featurs=32, out_features=32)
        # 5.linear2 torch.nn.Linear(in_featurs=32, out_features=32)

        # 6 Conv1d
        # torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=2, padding=0, dilation=1, groups=1, padding_mode='zeros')
        conv1d_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=2)
        self.conv1d_2 = conv1d_2

        # 7 BatchNorm1d
        # torch.nn.BatchNorm1d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        batch_norm1d_2 = BatchNorm1d(32)
        self.batch_norm1d_2 = batch_norm1d_2

        # 8 ReLU
        # relu_2 = torch.nn.ReLU()
        # self.relu_2 = relu_2

        # 9 ConvBlock(in_channels=32, out_channels_pw_proj=32, has_out_proj=False)
        conv_block_3 = ConvBlock(in_channels=32, out_channels_pw_proj=32, has_out_proj=False)
        self.conv_block_3 = conv_block_3

        # 10 TransformerLayer
        # att 96
        transformer_layer_3 = TransformerLayer(shape=32, att_qkv_in=32, att_qkv_out=96, scale=4.0)
        self.transformer_layer_3 = transformer_layer_3

        # 11 Conv1d
        # torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros')
        conv1d_3 =  nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.conv1d_3 = conv1d_3

        # 12 BatchNorm1d
        # torch.nn.BatchNorm1d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        batch_norm1d_3 = BatchNorm1d(32)
        self.batch_norm1d_3 = batch_norm1d_3

        # 13
        # relu_3 = torch.nn.ReLU()
        # self.relu_3 = relu_3

        # 14 ConvBlock
        conv_block_4 = ConvBlock(in_channels=32, out_channels_pw_proj=64, has_out_proj=True)
        self.conv_block_4 = conv_block_4

        # 15 TransformerLayer
        # att 192
        transformer_layer_4 = TransformerLayer(shape=64, att_qkv_in=64, att_qkv_out=192, scale=4 * np.sqrt(2))
        self.transformer_layer_4 = transformer_layer_4

        # 16 Conv1d
        conv1d_4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv1d_4 = conv1d_4

        # 17 BatchNorm1d
        batch_norm1d_4 = BatchNorm1d(64)
        self.batch_norm1d_4 = batch_norm1d_4

        # 18 ReLU
        # relu_4 = torch.nn.ReLU()
        # self.relu_4 = relu_4

    def load_state_dict(self, state_dict, prefix=''):
        prefix += 'sequential.'
        mapping = {
            # x0 = self.transformer(x)
            '0.': 'transformer.',
            # x1 = self.conv1d_1(x0)
            '1.': 'conv1d_1.',
            # x2 = self.batch_norm1d_1(x1)
            '2.': 'batch_norm1d_1.',
            # x3 = x2.relu()

            # x4 = self.conv_block_1(x3)
            '4.dw_conv.0.': 'conv_block_1.dw_conv.',
            '4.pw_conv.0.': 'conv_block_1.pw_conv.',
            '4.proj.': 'conv_block_1.proj.',
            # x5 = self.transformer_layer_1(x4)
            # x5 = self.transformer_layer_1(x4)
            '5.': 'transformer_layer_1.',
            # x6 = self.conv1d_2(x5)
            '6.': 'conv1d_2.',
            # x7 = self.batch_norm1d_2(x6)
            '7.': 'batch_norm1d_2.',
            # x8 = x7.relu()

            # x9 = self.conv_block_3(x8)
            # x10 = self.transformer_layer_3(x9)
            # x11 = self.conv1d_3(x10)
            # x12 = self.batch_norm1d_3(x11)
            '9.dw_conv.0.': 'conv_block_3.dw_conv.',
            '9.pw_conv.0.': 'conv_block_3.pw_conv.',
            '9.proj.': 'conv_block_3.proj.',
            '10.': 'transformer_layer_3.',
            '11.': 'conv1d_3.',
            '12.': 'batch_norm1d_3.',
            # x13 = x12.relu()

            # x14 = self.conv_block_4(x13)
            # x15 = self.transformer_layer_4(x14)
            # x16 = self.conv1d_4(x15)
            # x17 = self.batch_norm1d_4(x16)
            '14.dw_conv.0.': 'conv_block_4.dw_conv.',
            '14.pw_conv.0.': 'conv_block_4.pw_conv.',
            '14.proj.': 'conv_block_4.proj.',
            '15.': 'transformer_layer_4.',
            '16.': 'conv1d_4.',
            '17.': 'batch_norm1d_4.',
            # x18 = x17.relu()
        }
        t = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                k = k.replace(prefix, '')
                for p in mapping:
                    if k.startswith(p):
                        k = k.replace(p, mapping[p])
                if 'num_batches_tracked' in k:
                    v = v.reshape((1,))
                t[k] = v

        # print('\n'.join(list(t.keys())))
        nn.state.load_state_dict(self, t)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.transformer(x)
        x1 = self.conv1d_1(x0)
        x2 = self.batch_norm1d_1(x1)
        x3 = x2.relu()

        x4 = self.conv_block_1(x3)
        x5 = self.transformer_layer_1(x4)
        x6 = self.conv1d_2(x5)
        x7 = self.batch_norm1d_2(x6)
        x8 = x7.relu()

        x9 = self.conv_block_3(x8)
        x10 = self.transformer_layer_3(x9)
        x11 = self.conv1d_3(x10)
        x12 = self.batch_norm1d_3(x11)
        x13 = x12.relu()

        x14 = self.conv_block_4(x13)
        x15 = self.transformer_layer_4(x14)
        x16 = self.conv1d_4(x15)
        x17 = self.batch_norm1d_4(x16)
        x18 = x17.relu()

        return x18

        # self.sequential = torch.nn.Sequential(transformer,
        #                                       conv1d_1,
        #                                       batch_norm1d_1,
        #                                       relu_1,

        #                                       conv_block_1,
        #                                       transformer_layer_1,
        #                                       conv1d_2,
        #                                       batch_norm1d_2,
        #                                       relu_2,

        #                                       conv_block_3,
        #                                       transformer_layer_3,
        #                                       conv1d_3,
        #                                       batch_norm1d_3,
        #                                       relu_3,

        #                                       conv_block_4,
        #                                       transformer_layer_4,
        #                                       conv1d_4,
        #                                       batch_norm1d_4,
        #                                       relu_4)

class LSTMCell:
  def __init__(self, input_size, hidden_size, dropout):
    self.dropout = dropout

    self.weights_ih = Tensor.uniform(hidden_size * 4, input_size)
    self.bias_ih = Tensor.uniform(hidden_size * 4)
    self.weights_hh = Tensor.uniform(hidden_size * 4, hidden_size)
    self.bias_hh = Tensor.uniform(hidden_size * 4)

  def __call__(self, x: Tensor, hc: Tensor) -> Tensor:
    # print(x.shape)
    gates = x.linear(self.weights_ih.T, self.bias_ih) + hc[:x.shape[0]].linear(self.weights_hh.T, self.bias_hh)

    i, f, g, o = gates.chunk(4, 1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()

    c = (f * hc[x.shape[0]:]) + (i * g)
    h = (o * c.tanh()).dropout(self.dropout)
    # print(c.shape, h.shape)
    newhc = Tensor.cat(h, c)
    # print(newhc.shape)
    return newhc


class LSTM:
    def __init__(self, input_size, hidden_size, layers, dropout):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers

        self.cells = [LSTMCell(input_size, hidden_size, dropout) if i == 0 else LSTMCell(hidden_size, hidden_size, dropout if i != layers - 1 else 0) for i in range(layers)]

    # TODO(irwin): _asdict?
    def load_state_dict(self, state_dict, prefix=''):
        mapping = {
            "weight_ih_l0": "cells.0.weights_ih",
            "weight_hh_l0": "cells.0.weights_hh",
            "bias_ih_l0": "cells.0.bias_ih",
            "bias_hh_l0": "cells.0.bias_hh",
            "weight_ih_l1": "cells.1.weights_ih",
            "weight_hh_l1": "cells.1.weights_hh",
            "bias_ih_l1": "cells.1.bias_ih",
            "bias_hh_l1": "cells.1.bias_hh"
        }
        t = {mapping[k.replace(prefix, '')]: v for k, v in state_dict.items() if k.startswith(prefix)}
        # print('\n'.join(t.keys()))
        nn.state.load_state_dict(self, t)


    # @TinyJit
    def __call__(self, x: Tensor, hc: Tensor) -> tuple[Tensor, Tensor]:
        # @TinyJit
        def _do_step(x_: Tensor, hc_: Tensor) -> Tensor:
            return self.do_step(x_, hc_)

        if hc is None:
            hc = Tensor.zeros(self.layers, 2 * x.shape[1], self.hidden_size, requires_grad=False)

        output = None
        for t in range(x.shape[0]):
            # hc = self.do_step(x[t], hc) # TODO: why do we need to do this?
            hc = _do_step(x[t] + 1 - 1, hc) # TODO: why do we need to do this?
            if output is None:
                output = hc[-1:, :x.shape[1]]
            else:
                output = output.cat(hc[-1:, :x.shape[1]], dim=0)
        print(output.shape)
        return output.realize(), hc.realize()

    def do_step(self, x: Tensor, hc: Tensor) -> Tensor:
        print(x.shape)
        new_hc = [x]
        for i, cell in enumerate(self.cells):
            new_hc.append(cell(new_hc[i][:x.shape[0]], hc[i]))

        stacked = Tensor.stack(new_hc[1:])
        # print(stacked.shape)
        return stacked

class Decoder:
    def __init__(self):
        # decoder.1.weight
        # decoder.1.bias

        self.conv1d = nn.Conv1d(in_channels=64, out_channels=2, kernel_size=1)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv1d(x.relu()).mean(axis=2, keepdim=True).sigmoid()
            # torch.nn.ReLU(),
            # torch.nn.Conv1d(in_channels=64, out_channels=2, kernel_size=1),
            # torch.nn.AdaptiveAvgPool1d(output_size=1),
            # torch.nn.Sigmoid()

    def load_state_dict(self, state_dict, prefix=''):
        t = {k.replace(prefix, 'conv1d.'): v for k, v in state_dict.items() if k.startswith(prefix)}
        # print(t)
        nn.state.load_state_dict(self, t)

class STFT:
    filter_length : int
    hop_length : int
    win_length : int
    window : str

    def __init__(self):
        self.filter_length: int = 256
        self.hop_length : int = 64
        self.win_length : int = 256
        self.window : str = "hann"

        self.forward_basis_buffer = Tensor.zeros(258, 1, 256)

    def forward(self, input_data: Tensor) -> Tensor:
        return self.transform_(input_data)

    def transform_(self, input_data: Tensor) -> Tensor:
        filter_length = self.filter_length

        num_batches = input_data.shape[0]
        num_samples = input_data.shape[1]

        # [N, 1536] -> [N, 1, 1536]
        for_padding = input_data.reshape([num_batches, 1, num_samples])
        half = filter_length // 2
        # padded = for_padding.pad([half, half, 0, 0], "reflect", 0.)
        padded = simple_pad(for_padding, half)
        # [N, 1, 1536] -> [N, 1, 1, 1, 1536]
        padded_squeezed = padded
        forward_transform = padded_squeezed.conv2d(self.forward_basis_buffer, stride=self.hop_length)
        # print(forward_transform.shape)
        cutoff = half + 1

        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = Tensor.sqrt(real_part ** 2 + imag_part ** 2)
        return magnitude

# 1d pad with reflect across last dim a tensor of shape [N, 1, S]
def simple_pad(x: Tensor, pad: int) -> Tensor:
    left_pad = x[:, :, 1: 1+pad].flip(-1)
    right_pad = x[:, :, -1 - pad: -1].flip(-1)
    return Tensor.cat(left_pad, x, right_pad, dim=2)

class AdaptiveAudioNormalization:
    filter_: Tensor
    to_pad: int

    def __init__(self):
        self.to_pad = 3

        self.filter_ = Tensor.zeros(1, 1, 7)

    def forward(self, spect: Tensor) -> Tensor:
        megabyte: int = 1024 * 1024
        spect = (spect * megabyte + 1.0).log()
        if len(spect.shape) == 2:
            spect = spect[None, :, :]
        mean = spect.mean(1, keepdim=True)
        mean = simple_pad(mean, self.to_pad)
        mean = mean.conv2d(self.filter_)
        mean_mean = mean.mean(-1, keepdim=True)
        spect = spect.add(-mean_mean)
        return spect

class Silero:
    def __init__(self):
        self.feature_extractor = STFT()
        self.adaptive_normalization = AdaptiveAudioNormalization()
        self.first_layer = ConvBlock()
        self.encoder = Encoder()
        self.lstm = LSTM(64, 64, 2, 0.1)
        self.decoder = Decoder()

    def load_state_dict(self, silero_state_dict_tg):
        load_state_dict_prefix(self.feature_extractor, silero_state_dict_tg, prefix="feature_extractor.")
        load_state_dict_prefix(self.adaptive_normalization, silero_state_dict_tg, prefix="adaptive_normalization.")
        load_state_dict_prefix(self.first_layer, silero_state_dict_tg, prefix="first_layer.")
        load_state_dict_prefix(self.encoder, silero_state_dict_tg, prefix="encoder.")
        load_state_dict_prefix(self.lstm, silero_state_dict_tg, "lstm.")
        load_state_dict_prefix(self.decoder, silero_state_dict_tg, "decoder.1.")

    def __call__(self, x: Tensor, *args) -> Tensor:
        return self.forward(x, *args)

    # @TinyJit
    def forward(self, x: Tensor, h: Tensor, c: Tensor):
        if False:
            if hc is None:
                # [layers, batch, features]
                # h [2, 1, 64]
                #       +
                # c [2, 1, 64]
                hc = Tensor.zeros(2, 2, 64)
        hc = Tensor.cat(h, c, dim=1)

        @TinyJit
        def not_lstm(x):
            x = self.feature_extractor.forward(x)
            x = self.adaptive_normalization.forward(x)
            x = self.first_layer(x)
            x = self.encoder(x)
            return x.realize()

        with Timing("not lstm: "):
            x = not_lstm(x)
            # print(x.shape)

        # with Timing("adaptive_normalization: "):
        # with Timing("first_layer: "):
        # with Timing("encoder: "):

        # (batch, feature, seq) - > (seq, batch, feature)
        test_batch = False
        if test_batch:
            x = x.cat(x)

        batch_size = x.shape[0]

        hctg = hc

        with Timing("lstm: "):
            if batch_size > 1:
                if True:
                    # NOTE(irwin): batch support for sequential chunks, unbatches lstm processing (untested with batch_size > 1 yet)
                    res = []
                    batches = x
                    for batch in range(batch_size):
                        permx = batches[batch].unsqueeze(0).permute([2, 0, 1])
                        # print(permx.shape)
                        x = self.lstm(permx, hctg)
                        x, hc = x[0], x[1]
                        res.append(x)
                        hctg = hc
                    x = Tensor.cat(*res, dim=1)
                else:
                    # res = []
                    print(f"x.shape: {x.shape}")
                    batches = x.reshape(1, 64, -1) # [B, 64, 7]
                    print(f"batches.shape: {batches.shape}")
                    print(f"hctg.shape: {hctg.shape}")
                    x = self.lstm(x.permute([2, 0, 1]), hctg)
                    x, hc = x[0], x[1]
                    # for batch in range(batch_size):
                    #     permx = batches[batch].unsqueeze(0).permute([2, 0, 1])
                    #     # print(permx.shape)
                    #     x = self.lstm(permx, hctg)
                    #     x, hc = x[0], x[1]
                    #     res.append(x)
                    #     hctg = hc
                    # x = Tensor.cat(*res, dim=1)
            else:
                x = self.lstm(x.permute([2, 0, 1]), hctg)
                x, hc = x[0], x[1]

        # decoder [7, 1, 64] -> [1, 64, 7] (seq, batch, feature) -> (batch, feature, seq)
        x = x.permute([1, 2, 0])

        if test_batch:
            hc = hc.permute(1, 2, 0)
            hc1 = hc[:2]
            hc2 = hc[2:4]
            assert np.equal(x[0], x[1])
            assert np.equal(hc1, hc2)

        with Timing("decoder: "):
            x = self.decoder(x)
        # x = x[0].permute([1, 2, 0])
        # x = x.relu()
        # x = decoder.conv1d(x)
        # x = x.mean(axis=2, keepdim=True)
        # x = x.sigmoid()

        decoder_out_tg = x

        return decoder_out_tg.realize(), hc[:, 0, None, :].realize(), hc[:, 1, None, :].realize()

def foo():
    from pathlib import Path

    silero = Silero()
    silero.load_state_dict(nn.state.torch_load('silero_vad_v3_16k.pt'))

    audio_data = np.fromfile(r'RED.s16le', dtype=np.int16)
    torch_probs = np.loadtxt("torch.probs")

    h0 = np.zeros((2,1,64), dtype=np.float32)
    c0 = np.zeros_like(h0)

    hn = Tensor(h0)
    cn = Tensor(c0)

    batch_size = 64
    # example random input [batch_size, 1536]
    # rand_input = torch.rand([batch_size, 1536])

    # silero = torch.jit.script(silero_restored2, example_inputs=(rand_input, hn, cn))
    # silero_stateless = torch.jit.trace(silero_restored2.forward_stateless, example_inputs=(rand_input, ))
    # silero_stateful = torch.jit.trace(silero_restored2.forward_stateful, example_inputs=(torch.rand((1, 64, 7)), hn, cn))

    probs = []
    for chunk_batch in chunks_grouped(audio_data, batch_size):
        result = silero.forward(Tensor(np.array([c for c in chunk_batch if c is not None])), hn, cn)

        prob_result = result[0]
        hn = result[1]
        cn = result[2]

        for n in range(prob_result.shape[0]):
            prob = prob_result[n][1].item()
            probs.append(prob)

    if True:
        Path('current.probs').write_text('\n'.join(f"{p:0.6f}" for p in probs))
        print(np.abs(probs - torch_probs).mean())
        # print(subprocess.check_output(["fc", "current.probs", "torch.probs"]).decode('utf-8', errors='replace'))

    if False:
        probs = []
        for i, chunk in enumerate(chunks(audio_data)):
            result = jit_model._model1(torch.from_numpy(chunk), hn, cn).item()
            probs.append(result)
            break

if __name__ == "__main__":
    Tensor.training = False
    foo()