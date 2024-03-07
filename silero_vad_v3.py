import torch
from pathlib import Path

import sys
import tqdm
import numpy as np

from types import NoneType
from typing import Optional, Tuple
from torch import Value, ops
import torch.nn.functional
import torch.nn.functional as F

torch.set_grad_enabled(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def chunks(audio_data: np.ndarray, chunk_size=1536, window_size_in_chunks=96):
    # window_size_in_chunks = 96
    # chunk_size = 1536
    window_size_in_samples = window_size_in_chunks * chunk_size

    # h0, c0 = np.zeros((2, 1, 64), dtype=np.float32), np.zeros((2, 1, 64), dtype=np.float32)
    # hn, cn = h0, c0
    # all_probs = []
    for i in tqdm.tqdm(range(0, len(audio_data), window_size_in_samples)):
        window_s16 = audio_data[i:i+window_size_in_samples]
        window = window_s16.astype(np.float32)
        max_value = np.max(np.abs(window))
        window_normalized = window / max_value if max_value > 0 else window

        # window_probs = []
        for j in range(0, len(window_normalized), chunk_size):
            chunk = window_normalized[j:j+chunk_size]
            if len(chunk) < chunk_size:
                print("Warning: chunk is smaller than chunk size", file=sys.stderr)
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
            yield chunk

            # input = np.reshape(chunk, (1, chunk_size))
            # outputs = session.run([input, hn, cn])
            # hn, cn = outputs[1], outputs[2]
            # prob_output = outputs[0]
            # prob = prob_output[0][1][0]
            # window_probs.append(prob)

        # print('\n'.join(f"{n:0.6f}" for n in window_probs))
        # all_probs.extend(window_probs)

from types import NoneType
from typing import Optional, Tuple
from torch import ops
import torch.nn.functional


def transform_(self, input_data):
    num_batches = input_data.size(0)
    num_samples = input_data.size(1)

    input_data = input_data.view(num_batches, 1, num_samples)
    input_data = torch.nn.functional.pad(
        input_data.unsqueeze(1),
        (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
        mode='reflect')
    input_data = input_data.squeeze(1)
    forward_transform = torch.nn.functional.conv1d(
        input_data,
        self.forward_basis_buffer,
        stride=self.hop_length,
        padding=0)
    cutoff = int((self.filter_length / 2) + 1)
    real_part = forward_transform[:, :cutoff, :]
    imag_part = forward_transform[:, cutoff:, :]
    magnitude = torch.sqrt(real_part**2 + imag_part**2)
    phase = torch.atan2(imag_part.data, real_part.data)
    return magnitude, phase


class STFT(torch.nn.Module):
    filter_length : int
    hop_length : int
    win_length : int
    window : str

    def __init__(self):
        super().__init__()

        self.filter_length: int = 256
        self.hop_length : int = 64
        self.win_length : int = 256
        self.window : str = "hann"

        self.register_buffer("forward_basis_buffer", torch.zeros([258, 1, 256]))

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        _0 = (self).transform_(input_data, )
        return _0

    def transform_(self, input_data: torch.Tensor) -> torch.Tensor:
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
        input_data0 = input_data.view([num_batches, 1, num_samples])
        _1 = torch.unsqueeze(input_data0, 1)
        filter_length = self.filter_length
        filter_length_half = int(filter_length / 2)
        input_data1 = torch.nn.functional.pad(_1, [filter_length_half, filter_length_half, 0, 0], "reflect", 0., )
        input_data2 = torch.squeeze(input_data1, 1)
        forward_basis_buffer = self.forward_basis_buffer
        hop_length = self.hop_length
        forward_transform = torch.conv1d(input_data2, forward_basis_buffer, None, [hop_length], [0])
        # filter_length1 = self.filter_length
        # _4 = torch.add(torch.div(filter_length1, 2), 1)
        # cutoff = int(_4)
        cutoff = filter_length_half + 1
        # _5 = torch.slice(torch.slice(forward_transform), 1, None, cutoff)
        # real_part = torch.slice(_5, 2)
        real_part = forward_transform[:, :cutoff, :]

        # _6 = torch.slice(torch.slice(forward_transform), 1, cutoff)
        # imag_part = torch.slice(_6, 2)
        imag_part = forward_transform[:, cutoff:, :]

        _7 = torch.add(torch.pow(real_part, 2), torch.pow(imag_part, 2))
        magnitude = torch.sqrt(_7)
        # phase = torch.atan2(imag_part.data, real_part.data)
        # return (magnitude, phase)
        return magnitude

def simple_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    left_pad = torch.flip(x[:, :, 1: 1+pad], (-1,))
    right_pad = torch.flip(x[:, :, -1 - pad: -1], (-1,))
    return torch.cat([left_pad, x, right_pad], dim=2)

class AdaptiveAudioNormalization(torch.nn.Module):
    filter_: torch.Tensor
    to_pad: int

    def __init__(self):
        super().__init__()

        self.to_pad = 3

        self.register_buffer("filter_", torch.zeros((1, 1, 7)))

    def forward(self, spect: torch.Tensor) -> torch.Tensor:
        spect = torch.log1p(spect * 1048576)
        if len(spect.shape) == 2:
            spect = spect[None, :, :]
        mean = spect.mean(dim=1, keepdim=True)
        mean = simple_pad(mean, self.to_pad)
        mean = torch.conv1d(mean, self.filter_)
        mean_mean = mean.mean(dim=-1, keepdim=True)
        spect = spect.add(-mean_mean)
        return spect

class ConvBlock(torch.nn.Module):
    # __parameters__ = []
    # __buffers__ = []
    # training : bool
    # _is_full_backward_hook : Optional[bool]
    # dw_conv : __torch__.torch.nn.modules.container.Sequential
    # pw_conv : __torch__.torch.nn.modules.container.___torch_mangle_1.Sequential
    # proj : __torch__.torch.nn.modules.conv.___torch_mangle_0.Conv1d

    def __init__(self, in_channels: int = 129, out_channels_pw_proj: int = 16, has_out_proj: bool = True):
        super().__init__()
        # maybe equivalent to put in forward this: torch.conv1d(input, weight, bias, [1], [2], [1], 129)
        self.dw_conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=1, padding=2, dilation=1, groups=in_channels, padding_mode='zeros'),
            torch.nn.Identity(),
            torch.nn.ReLU()
        )

        # maybe equivalent to put in forward this: torch.conv1d(input, weight, bias, [1], [0], [1], 1)
        self.pw_conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels_pw_proj, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'),
            torch.nn.Identity()
        )
        if has_out_proj:
            # maybe equivalent to put in forward this: torch.conv1d(input, weight, bias, [1], [0], [1])
            self.proj = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels_pw_proj, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros')
        else:
            # TODO(irwin): consider identity to get rid of if check in forward
            self.proj = torch.nn.Identity()
        self.activation = torch.nn.ReLU()

    # def forward_(self: __torch__.models.number_vad_model.ConvBlock, x: torch.Tensor) -> torch.Tensor:
    #     pw_conv = self.pw_conv
    #     dw_conv = self.dw_conv
    #     x8 = (pw_conv).forward((dw_conv).forward(x, ), )
    #     proj = self.proj
    #     residual = (proj).forward(x, )
    #     x9 = torch.add_(x8, residual)
    #     activation = self.activation
    #     return (activation).forward(x9, )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pw_conv(self.dw_conv(x))
        x += self.proj(residual)
        x = self.activation(x)
        return x


class MultiHeadAttention(torch.nn.Module):
    scale : float
    n_heads : int
    has_out_proj : bool
#   QKV : __torch__.torch.nn.modules.linear.Linear
#   out_proj : __torch__.torch.nn.modules.linear.___torch_mangle_3.Linear

    def __init__(self, qkv_in_features: int, qkv_out_features: int, scale: float = 2 * np.sqrt(2), n_heads: int = 2):
        super().__init__()
        self.scale = scale
        self.n_heads = n_heads

        self.QKV = torch.nn.Linear(in_features=qkv_in_features, out_features=qkv_out_features)
        self.has_out_proj = True

        if self.has_out_proj:
            self.out_proj = torch.nn.Linear(in_features=qkv_in_features, out_features=qkv_in_features)
        else:
            self.out_proj = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # _1 = __torch__.torch.nn.functional.softmax

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

        k = k.transpose(0, 1).contiguous().view(seq, bsz * self.n_heads, head_dim).transpose(0, 1)

        # _3 = torch.contiguous(torch.transpose(k, 0, 1))
        # n_heads0 = self.n_heads
        # _4 = [seq, torch.mul(bsz, n_heads0), head_dim]
        # k0 = torch.transpose(torch.view(_3, _4), 0, 1)


        q = q.transpose(0, 1).contiguous().view(seq, bsz * self.n_heads, head_dim).transpose(0, 1)

        # _5 = torch.contiguous(torch.transpose(q, 0, 1))
        # n_heads1 = self.n_heads
        # _6 = [seq, torch.mul(bsz, n_heads1), head_dim]
        # q0 = torch.transpose(torch.view(_5, _6), 0, 1)

        v = v.transpose(0, 1).contiguous().view(seq, bsz * self.n_heads, head_dim).transpose(0, 1)

        # _7 = torch.contiguous(torch.transpose(v, 0, 1))
        # n_heads2 = self.n_heads
        # _8 = [seq, torch.mul(bsz, n_heads2), head_dim]
        # v0 = torch.transpose(torch.view(_7, _8), 0, 1)

        alpha = F.softmax(k @ q.transpose(1, 2) / self.scale, dim=-1)  # (bs * head, seq, hid/head) @ (bs / head, hid / head, seq)
        # _9 = torch.matmul(k, torch.transpose(q, 1, 2))
        # scale = self.scale
        # alpha = _1(torch.div(_9, scale), -1, 3, None, )

        attn = alpha @ v  # (bs * head, seq, seq) @ (bs * head, seq, hid / head)
        # attn = torch.matmul(alpha, v)

        # (bs * head, seg, hid / head) -> (seq, bs * head, hid / head) ->  (seq, bs, hid) ->  (bs, seq, hid)
        attn = attn.transpose(0, 1).contiguous().view(seq, bsz, dim).transpose(0, 1)
        # _10 = torch.contiguous(torch.transpose(attn, 0, 1))
        # attn0 = torch.transpose(torch.view(_10, [seq, bsz, dim]), 0, 1)

        # if self.has_out_proj:
        #     attn = self.out_proj(attn)
        attn = self.out_proj(attn)

        return attn

        # has_out_proj = self.has_out_proj
        # if has_out_proj:
        #   out_proj = self.out_proj
        #   attn1 = (out_proj).forward(attn0, )
        # else:
        #   attn1 = attn0
        # return attn1


class TransformerLayer(torch.nn.Module):
    # reshape_inputs : bool
    training : bool

    def __init__(self, shape: int, att_qkv_in: int, att_qkv_out: int, scale: float = 2 * np.sqrt(2)):
        super().__init__()
        # self.reshape_inputs : bool = True
        self.training : bool = False

        self.attention = MultiHeadAttention(qkv_in_features=att_qkv_in, qkv_out_features=att_qkv_out, scale=scale)
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
        # if self.reshape_inputs:
        #     x = x.permute(0, 2, 1).contiguous()
        x = x.permute(0, 2, 1).contiguous()

        attn = self.attention(x)
        x = x + self.dropout1(attn)
        x = self.norm1(x)

        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        # (batch * sequence * dims) => (batch * dims * sequence)
        # if self.reshape_inputs:
        #     x = x.permute(0, 2, 1).contiguous()
        x = x.permute(0, 2, 1).contiguous()
        return x

    # def forward__(self, x: torch.Tensor) -> torch.Tensor:
    #     reshape_inputs = self.reshape_inputs
    #     if reshape_inputs:
    #         x1 = torch.contiguous(torch.permute(x, [0, 2, 1]))
    #         x0 = x1
    #     else:
    #         x0 = x
    #     attention = self.attention
    #     attn = (attention).forward(x0, )
    #     dropout1 = self.dropout1
    #     x2 = torch.add(x0, (dropout1).forward(attn, ))
    #     norm1 = self.norm1
    #     x3 = (norm1).forward(x2, )
    #     linear2 = self.linear2
    #     dropout = self.dropout
    #     activation = self.activation
    #     linear1 = self.linear1
    #     _0 = (activation).forward((linear1).forward(x3, ), )
    #     x20 = (linear2).forward((dropout).forward(_0, ), )
    #     dropout2 = self.dropout2
    #     x4 = torch.add(x3, (dropout2).forward(x20, ))
    #     norm2 = self.norm2
    #     x5 = (norm2).forward(x4, )
    #     reshape_inputs0 = self.reshape_inputs
    #     if reshape_inputs0:
    #         x7 = torch.contiguous(torch.permute(x5, [0, 2, 1]))
    #         x6 = x7
    #     else:
    #         x6 = x5
    #     return x6

    # def forward_(self, x: torch.Tensor) -> torch.Tensor:
    #     reshape_inputs = self.reshape_inputs
    #     if reshape_inputs:
    #         x1 = torch.contiguous(torch.permute(x, [0, 2, 1]))
    #         x0 = x1
    #     else:
    #         x0 = x
    #     attention = self.attention
    #     attn = (attention).forward(x0, )
    #     dropout1 = self.dropout1
    #     x2 = torch.add(x0, (dropout1).forward(attn, ))
    #     norm1 = self.norm1
    #     x3 = (norm1).forward(x2, )
    #     linear2 = self.linear2
    #     dropout = self.dropout
    #     activation = self.activation
    #     linear1 = self.linear1
    #     _0 = (activation).forward((linear1).forward(x3, ), )
    #     x20 = (linear2).forward((dropout).forward(_0, ), )
    #     dropout2 = self.dropout2
    #     x4 = torch.add(x3, (dropout2).forward(x20, ))
    #     norm2 = self.norm2
    #     x5 = (norm2).forward(x4, )
    #     reshape_inputs0 = self.reshape_inputs
    #     if reshape_inputs0:
    #         x7 = torch.contiguous(torch.permute(x5, [0, 2, 1]))
    #         x6 = x7
    #     else:
    #         x6 = x5
    #     return x6

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # att 48
        transformer = TransformerLayer(shape=16, att_qkv_in=16, att_qkv_out=48, scale=2 * np.sqrt(2))
        # self.transformer = transformer

        # full: in_channels=16, out_channels=16, kernel_size=1, stride=2, padding=0, dilation=1, groups=1, padding_mode='zeros'
        conv1d_1 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=2)
        # self.conv1d_1 = conv1d_1

        # full: num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        batch_norm1d_1 = torch.nn.BatchNorm1d(16)
        # self.batch_norm1d_1 = batch_norm1d_1
        relu_1 = torch.nn.ReLU()
        # self.relu_1 = relu_1

        # 4.0, ConvBlock
        conv_block_1 = ConvBlock(in_channels=16, out_channels_pw_proj=32)
        # self.conv_block_1 = conv_block_1

        # 5 TransformerLayer
        # att 96
        transformer_layer_1 = TransformerLayer(shape=32, att_qkv_in=32, att_qkv_out=96, scale=4.0)
        # self.transformer_layer_1 = transformer_layer_1
        # 5.attention MultiHeadAttention(in_features=32, scale=4.0)
        # 5.norm1 torch.nn.LayerNorm(normalized_shape=32)
        # 5.norm2 torch.nn.LayerNorm(normalized_shape=32)
        # 5.linear1 torch.nn.Linear(in_featurs=32, out_features=32)
        # 5.linear2 torch.nn.Linear(in_featurs=32, out_features=32)

        # 6 Conv1d
        # torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=2, padding=0, dilation=1, groups=1, padding_mode='zeros')
        conv1d_2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=2)
        # self.conv1d_2 = conv1d_2

        # 7 BatchNorm1d
        # torch.nn.BatchNorm1d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        batch_norm1d_2 = torch.nn.BatchNorm1d(32)
        # self.batch_norm1d_2 = batch_norm1d_2

        # 8 ReLU
        relu_2 = torch.nn.ReLU()
        # self.relu_2 = relu_2

        # 9 ConvBlock(in_channels=32, out_channels_pw_proj=32, has_out_proj=False)
        conv_block_3 = ConvBlock(in_channels=32, out_channels_pw_proj=32, has_out_proj=False)
        # self.conv_block_3 = conv_block_3

        # 10 TransformerLayer
        # att 96
        transformer_layer_3 = TransformerLayer(shape=32, att_qkv_in=32, att_qkv_out=96, scale=4.0)
        # self.transformer_layer_3 = transformer_layer_3

        # 11 Conv1d
        # torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros')
        conv1d_3 =  torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        # self.conv1d_3 = conv1d_3

        # 12 BatchNorm1d
        # torch.nn.BatchNorm1d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        batch_norm1d_3 = torch.nn.BatchNorm1d(num_features=32)
        # self.batch_norm1d_3 = batch_norm1d_3

        # 13
        relu_3 = torch.nn.ReLU()
        # self.relu_3 = relu_3

        # 14 ConvBlock
        conv_block_4 = ConvBlock(in_channels=32, out_channels_pw_proj=64, has_out_proj=True)
        # self.conv_block_4 = conv_block_4

        # 15 TransformerLayer
        # att 192
        transformer_layer_4 = TransformerLayer(shape=64, att_qkv_in=64, att_qkv_out=192, scale=4 * np.sqrt(2))
        # self.transformer_layer_4 = transformer_layer_4

        # 16 Conv1d
        conv1d_4 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        # self.conv1d_4 = conv1d_4

        # 17 BatchNorm1d
        batch_norm1d_4 = torch.nn.BatchNorm1d(num_features=64)
        # self.batch_norm1d_4 = batch_norm1d_4

        # 18 ReLU
        relu_4 = torch.nn.ReLU()
        # self.relu_4 = relu_4

        self.sequential = torch.nn.Sequential(transformer,
                                              conv1d_1,
                                              batch_norm1d_1,
                                              relu_1,

                                              conv_block_1,
                                              transformer_layer_1,
                                              conv1d_2,
                                              batch_norm1d_2,
                                              relu_2,

                                              conv_block_3,
                                              transformer_layer_3,
                                              conv1d_3,
                                              batch_norm1d_3,
                                              relu_3,

                                              conv_block_4,
                                              transformer_layer_4,
                                              conv1d_4,
                                              batch_norm1d_4,
                                              relu_4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)

def print_interesting(jit_module):
    interesting_keys = [
        # 'original_name',

        'named_parameters',
        'named_buffers',
        'named_modules',
        'named_children',

        # 'code',
        # 'graph',

        # 'state_dict',

        # 'children',
        # 'inlined_graph',
        # 'modules',
        # 'parameters',
    ]

    print(f"original_name {jit_module.original_name}")
    for key in interesting_keys:
        val = getattr(jit_module, key)
        dicted = {k: v for k,v in val() if k}
        if len(dicted):
            print(f"{key}:")
            for subkey, subval in val():
                if subkey:
                    shortstr = str(subval)[:80].replace('\n','') if str(subval).startswith('tensor') else str(subval)
                    print(f"  {subkey}:\t{shortstr}")

    leftover_keys = set(dir(jit_module)) - set(dir(type(jit_module)))
    # print(leftover_keys)
    for key in leftover_keys:
        if not key.startswith('_'):
            print(key, getattr(jit_module, key))

    state_keys = '\n'.join(jit_module.state_dict().keys())
    if state_keys:
        print(f"state_dict keys:\n  {state_keys}")

    print('\nCODE\n====')
    print(jit_module.code)
    print('\nGRAPH\n====')
    print(jit_module.graph)
    print('\nINLINE GRAPH\n====')
    print(jit_module.inlined_graph)

def lookup(module, path):
    for part in path.split('.'):
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            return None
    return module

def print_module_init_args_with_lookup(module, path):
    module = lookup(module, path)
    if module is not None and hasattr(torch.nn, module.original_name):
        print_module_init_args(module, getattr(torch.nn, module.original_name))

def print_conv1d_params(jit_module):
  keys = [
    "out_channels",
    "in_channels",
    "kernel_size",
    "output_padding",
    "dilation",
    "stride",
    "padding",
    "groups",
    "padding_mode",
    "transposed",
  ]
  for key in keys:
    val = getattr(jit_module, key)
    print(f"{key}:\t{val}")

# print_conv1d_params(getattr(encoder, '1'))

# def spam(x, eggs):
#   pass

# print(spam.__kwdefaults__)

def print_module_init_args(jit_module, original_module=torch.nn.Conv1d):
    init_co = original_module.__init__.__code__
    varnames = init_co.co_varnames[1:init_co.co_argcount]
    defaults = original_module.__init__.__defaults__
    if defaults is None:
        defaults = []
    defaults_dict = dict(list(zip(varnames[::-1], defaults[::-1]))[::-1])
    arglist = []
    for i, varname in enumerate(varnames):
        # if varname not in original_module.__init__.__defaults__
        attribute = getattr(jit_module, varname, None)
        if attribute is not None and not isinstance(attribute, torch.Tensor):
            if isinstance(attribute, tuple) and len(attribute) == 1:
               attribute = attribute[0]
            arglist.append((varname, attribute))
            # print(varname, attribute)
    def helper(args):
        print(f"torch.nn.{original_module.__name__}({args})")
    # helper(', '.join(f"{k}={v!r}" for k, v in arglist))
    helper(', '.join(f"{k}={v!r}" for k, v in arglist if k not in defaults_dict or defaults_dict[k] != v))

class Silero_VAD_V3(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = STFT()
        self.adaptive_normalization = AdaptiveAudioNormalization()
        self.first_layer = ConvBlock()

        self.encoder = Encoder()
        self.lstm = torch.nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)

        self.decoder = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=64, out_channels=2, kernel_size=1),
            torch.nn.AdaptiveAvgPool1d(output_size=1),
            torch.nn.Sigmoid()
        )

    def spam(self, input, h, c) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # h, c = hc

        batch, seq, feat = input.shape
        r, (h, c) = self.lstm(input.reshape(1, -1, feat), (h, c))
        r = r.reshape(batch, seq, -1)

        # r0, (h, c) = self.lstm(input[0:0+1, :, :], (h, c))
        # r1, (h, c) = self.lstm(input[1:1+1, :, :], (h, c))
        # r2, (h, c) = self.lstm(input[2:2+1, :, :], (h, c))
        # r3, (h, c) = self.lstm(input[3:3+1, :, :], (h, c))
        # r4, (h, c) = self.lstm(input[4:4+1, :, :], (h, c))
        # r5, (h, c) = self.lstm(input[5:5+1, :, :], (h, c))
        # r6, (h, c) = self.lstm(input[6:6+1, :, :], (h, c))
        # r7, (h, c) = self.lstm(input[7:7+1, :, :], (h, c))
        # r8, (h, c) = self.lstm(input[8:8+1, :, :], (h, c))
        # r9, (h, c) = self.lstm(input[9:9+1, :, :], (h, c))
        # r10, (h, c) = self.lstm(input[10:10+1, :, :], (h, c))
        # r11, (h, c) = self.lstm(input[11:11+1, :, :], (h, c))
        # r12, (h, c) = self.lstm(input[12:12+1, :, :], (h, c))
        # r13, (h, c) = self.lstm(input[13:13+1, :, :], (h, c))
        # r14, (h, c) = self.lstm(input[14:14+1, :, :], (h, c))
        # r15, (h, c) = self.lstm(input[15:15+1, :, :], (h, c))
        # r16, (h, c) = self.lstm(input[16:16+1, :, :], (h, c))
        # r17, (h, c) = self.lstm(input[17:17+1, :, :], (h, c))
        # r18, (h, c) = self.lstm(input[18:18+1, :, :], (h, c))
        # r19, (h, c) = self.lstm(input[19:19+1, :, :], (h, c))
        # r20, (h, c) = self.lstm(input[20:20+1, :, :], (h, c))
        # r21, (h, c) = self.lstm(input[21:21+1, :, :], (h, c))
        # r22, (h, c) = self.lstm(input[22:22+1, :, :], (h, c))
        # r23, (h, c) = self.lstm(input[23:23+1, :, :], (h, c))
        # r24, (h, c) = self.lstm(input[24:24+1, :, :], (h, c))
        # r25, (h, c) = self.lstm(input[25:25+1, :, :], (h, c))
        # r26, (h, c) = self.lstm(input[26:26+1, :, :], (h, c))
        # r27, (h, c) = self.lstm(input[27:27+1, :, :], (h, c))
        # r28, (h, c) = self.lstm(input[28:28+1, :, :], (h, c))
        # r29, (h, c) = self.lstm(input[29:29+1, :, :], (h, c))
        # r30, (h, c) = self.lstm(input[30:30+1, :, :], (h, c))
        # r31, (h, c) = self.lstm(input[31:31+1, :, :], (h, c))

        # r = torch.stack([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31], dim=0).squeeze()

        return r, (h, c)

    def forward_stateless(self, input_data):
        x = input_data

        x0 = (self.feature_extractor).forward(x, )
        x1 = (self.adaptive_normalization).forward(x0, )

        # baseline_first_layer = (first_layer).forward(x1, )
        x2 = (self.first_layer).forward(x1, )

        # baseline_encoder = (encoder).forward(x2, )
        x3 = (self.encoder).forward(x2, )
        return x3

    def forward_stateful(self, encoded_data, h, c):
        x3 = encoded_data

        # _3 = (lstm).forward__0(torch.permute(x3, [0, 2, 1]), (h, c), )
        _3 = self.lstm(torch.permute(x3, [0, 2, 1]), (h, c), )

        x6, _4, = _3
        hn1, cn1, = _4
        x4, hn, cn = x6, hn1, cn1
        # x7 = (decoder).forward(torch.permute(x4, [0, 2, 1]), )
        x7 = (self.decoder).forward(torch.permute(x4, [0, 2, 1]), )

        return (x7, hn, cn)

    def forward__(self, input_data, h, c):
        return self.forward_stateful(self.forward_stateless(input_data), h, c)

    def forward_(self, input_data, h, c):
        x = input_data

        x0 = (self.feature_extractor).forward(x, )
        x1 = (self.adaptive_normalization).forward(x0, )

        # baseline_first_layer = (first_layer).forward(x1, )
        x2 = (self.first_layer).forward(x1, )

        # baseline_encoder = (encoder).forward(x2, )
        x3 = (self.encoder).forward(x2, )
        # assert np.allclose(baseline_encoder, x3)

        # _3 = (lstm).forward__0(torch.permute(x3, [0, 2, 1]), (h, c), )
        _3 = self.lstm(torch.permute(x3, [0, 2, 1]), (h, c) )

        x6, _4, = _3
        hn1, cn1, = _4
        x4, hn, cn = x6, hn1, cn1
        # x7 = (decoder).forward(torch.permute(x4, [0, 2, 1]), )
        x7 = (self.decoder).forward(torch.permute(x4, [0, 2, 1]), )

        return (x7, hn, cn)

    def forward(self, input_data, h, c):
        x = input_data

        x0 = (self.feature_extractor).forward(x, )
        x1 = (self.adaptive_normalization).forward(x0, )

        # baseline_first_layer = (first_layer).forward(x1, )
        x2 = (self.first_layer).forward(x1, )

        # baseline_encoder = (encoder).forward(x2, )
        x3 = (self.encoder).forward(x2, )
        # assert np.allclose(baseline_encoder, x3)

        # _3 = (lstm).forward__0(torch.permute(x3, [0, 2, 1]), (h, c), )
        _3 = self.spam(torch.permute(x3, [0, 2, 1]), h, c )

        x6, _4, = _3
        hn1, cn1, = _4
        x4, hn, cn = x6, hn1, cn1
        # x7 = (decoder).forward(torch.permute(x4, [0, 2, 1]), )
        # print(x4.shape)
        x7 = (self.decoder).forward(torch.permute(x4, [0, 2, 1]), )

        return (x7, hn, cn)

def chunks_grouped(audio_data, group_size):
    from itertools import zip_longest
    args = [iter(chunks(audio_data))] * group_size
    return zip_longest(*args)

class OnnxWrapper():
    def __init__(self, path):
        import onnxruntime
        # onnxruntime.SessionOptions
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(path, options, providers=['CPUExecutionProvider'])

        self.reset_states()

    def reset_states(self):
        self._h = np.zeros((2, 1, 64)).astype('float32')
        self._c = np.zeros((2, 1, 64)).astype('float32')

    def __call__(self, x, h, c):
        ort_inputs = {'input': x.numpy(), 'h0': h.numpy(), 'c0': c.numpy()}
        ort_outs = self.session.run(None, ort_inputs)
        out = ort_outs

        return out

def foo():
    silero_restored2 = Silero_VAD_V3()
    silero_restored2.load_state_dict(torch.load("silero_vad_v3_16k.pt"))
    silero_restored2.train(False)


    audio_data = np.fromfile(r'RED.s16le', dtype=np.int16)
    torch_probs = np.loadtxt("torch.probs")


    h0 = np.zeros((2,1,64), dtype=np.float32)
    c0 = np.zeros_like(h0)

    hn = torch.from_numpy(h0)
    cn = torch.from_numpy(c0)

    CUSTOM_BATCH = 32
    batch_size = CUSTOM_BATCH
    # example random input [batch_size, 1536]
    rand_input = torch.rand([batch_size, 1536])

    silero = silero_restored2
    # silero = torch.jit.script(silero_restored2, example_inputs=(rand_input, hn, cn))

    # onnx_exported_filename = f"silero_restored_v3.1_16k_v2_b{batch_size}.onnx"
    onnx_exported_filename = f"silero_restored_v3.1_16k_v2_dyn.onnx"
    if False:
        torch.onnx.export(silero_restored2,
                        (rand_input, hn, cn),
                        onnx_exported_filename,
                        input_names=["input", "h0", "c0"],
                        output_names=["output", "hn", "cn"],
                        dynamic_axes={
                            "input": [0],
                            "output": [0],
                        })
    ort = OnnxWrapper(onnx_exported_filename)
    # silero_stateless = torch.jit.trace(silero_restored2.forward_stateless, example_inputs=(rand_input, ))
    # silero_stateful = torch.jit.trace(silero_restored2.forward_stateful, example_inputs=(torch.rand((1, 64, 7)), hn, cn))

    probs = []
    for chunk_batch in chunks_grouped(audio_data, batch_size):
        result_batch = []
        if batch_size == 1:
            for chunk in chunk_batch:
                result_stateless = silero.forward_stateless(torch.from_numpy(chunk).reshape([1, 1536]))
                result_batch.append(result_stateless)
        elif batch_size == CUSTOM_BATCH:
            chunks_batched = np.array([c for c in chunk_batch if c is not None])
            rem = batch_size - chunks_batched.shape[0]
            if rem > 0:
                chunks_batched = np.concatenate([chunks_batched, np.zeros((rem, 1536), dtype=np.float32)])

            result_stateless = ort(torch.from_numpy(chunks_batched), hn, cn)
            # result_stateless = silero.forward(torch.from_numpy(chunks_batched), hn, cn)

            result = result_stateless
            prob_result = result[0]
            try:
                hn = torch.from_numpy(result[1])
                cn = torch.from_numpy(result[2])
            except:
                hn = result[1]
                cn = result[2]

            for i in range(prob_result.shape[0] - rem):
                prob = prob_result[i][1].item()
                probs.append(prob)

        else:
            chunks_batched = np.array([c for c in chunk_batch if c is not None])
            result_stateless = silero.forward_stateless(torch.from_numpy(chunks_batched))
            for j in range(result_stateless.shape[0]):
                # print(result_stateless.shape)
                result_batch.append(result_stateless[j:j+1, :, :])

        if batch_size == CUSTOM_BATCH:
            pass
        else:
            for result_stateless in result_batch:
                result = silero.forward_stateful(result_stateless, hn, cn)
                # result = restored_model1(torch.from_numpy(chunk).reshape([1, 1536]), hn, cn)
                prob_result = result[0]
                hn = result[1]
                cn = result[2]

                prob = prob_result[0][1].item()
                probs.append(prob)
                # break

    # print(chunks_batched.shape)
    # print(prob)

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
    foo()