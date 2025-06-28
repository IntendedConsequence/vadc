import math
import os
os.environ["LLVM_PATH"]="c:\\1PROGS\\llvm-16.0.2-windows-amd64-msvc17-msvcrt\\bin\\LLVM-C.dll"
import pathlib

import tinygrad
import tinygrad.nn as nn
import numpy as np

from tinygrad import Tensor, Device, TinyJit, dtypes, Variable
from tinygrad.helpers import tqdm, Context, getenv, Timing
# Device.DEFAULT = "CUDA"

# import torch
# import lovely_tensors
# lovely_tensors.monkey_patch()

# from silero_vad import Silero_Vad_5
# import utils


def stft(x:Tensor, weight:Tensor, n_fft=256, stride=128, pad=64)->Tensor:
    cutoff = int(n_fft // 2) + 1
    tg_stft_raw = x.pad((0, pad), "reflect").unsqueeze(1).conv2d(weight, stride=stride)
    tg_stft_out = (tg_stft_raw[:, :cutoff, :]**2 + tg_stft_raw[:, cutoff:, :]**2).sqrt()
    return tg_stft_out


def encoder(x, parameters):
    encoder_shapes = [
        [[128, 129, 3], 1, 1],
        [[64, 128, 3], 2, 1],
        [[64, 64, 3], 2, 1],
        [[128, 64, 3], 1, 1],
    ]

    for (shape, stride, pad), (weight, bias) in zip(encoder_shapes, parameters):
        x = x.conv2d(weight=weight, bias=bias, stride=(stride,), padding=(pad,)).relu()
    return x


class LSTM:
    def __init__(self, lstm_sd:dict|None=None):
        self.cell = tinygrad.nn.LSTMCell(128, 128)
        if lstm_sd is not None:
            self.cell.weight_ih = Tensor(lstm_sd['weight_ih_l0'].numpy())
            self.cell.weight_hh = Tensor(lstm_sd['weight_hh_l0'].numpy())
            self.cell.bias_ih = Tensor(lstm_sd['bias_ih_l0'].numpy())
            self.cell.bias_hh = Tensor(lstm_sd['bias_hh_l0'].numpy())

    def tg_lstm_jit(self, x:Tensor, hc:tuple[Tensor, Tensor])->tuple[Tensor, Tensor]:
        hn, cn = self.cell(x, hc)
        return hn, cn

    # @TinyJit
    def __call__(self, x:Tensor, hc:tuple[Tensor, Tensor]|None=None)->tuple[Tensor, Tensor]:
        if hc is None: hc = (Tensor.zeros(x.size(0), self.cell.weight_hh.size(1), dtype=x.dtype, device=x.device), Tensor.zeros(x.size(0), self.cell.weight_hh.size(1), dtype=x.dtype, device=x.device))
        bs, seq, feat = x.shape
        USE_CAT = True
        if not USE_CAT:
            res = Tensor.empty(bs, seq, feat)
        else:
            res = Tensor.empty(bs, 0, feat)
        # vi = Variable("seq", 0, seq-1)
        for i in range(seq):
            # vib = vi.bind(i)
            hc = self.tg_lstm_jit(x[:, i, :], hc)
            if not USE_CAT:
                res[:, i:i+1, :] = hc[0].unsqueeze(1)
            else:
                res = Tensor.cat(res, hc[0].unsqueeze(1), dim=1)
        return res, hc

class SileroV5:
    def __init__(self, v5=None):
        if v5 is not None:
            self._init_from_torch(v5)
        else:
            self.lstm = LSTM()

    def _init_from_torch(self, v5):
        self.forward_basis_buffer = Tensor(v5.stft.forward_basis_buffer.numpy())
        self.encoder_parameters = [[Tensor(wb.detach().numpy()) for wb in e.parameters()] for e in v5.encoder]
        self.lstm = LSTM(v5.decoder.rnn.state_dict())
        self.decoder_parameters = list(Tensor(w.detach().numpy()) for w in v5.decoder.decoder.parameters())

    @staticmethod
    def from_sd(sd, device=None):
        v5tg = SileroV5()
        v5tg.forward_basis_buffer = sd["forward_basis_buffer"].to(device)
        v5tg.encoder_parameters = [
            [sd["encoder_parameters.0.0"].to(device), sd["encoder_parameters.0.1"].to(device)],
            [sd["encoder_parameters.1.0"].to(device), sd["encoder_parameters.1.1"].to(device)],
            [sd["encoder_parameters.2.0"].to(device), sd["encoder_parameters.2.1"].to(device)],
            [sd["encoder_parameters.3.0"].to(device), sd["encoder_parameters.3.1"].to(device)],
        ]
        v5tg.lstm.cell.weight_ih = sd["lstm.cell.weight_ih"].to(device)
        v5tg.lstm.cell.weight_hh = sd["lstm.cell.weight_hh"].to(device)
        v5tg.lstm.cell.bias_ih = sd["lstm.cell.bias_ih"].to(device)
        v5tg.lstm.cell.bias_hh = sd["lstm.cell.bias_hh"].to(device)

        v5tg.decoder_parameters = [
            sd["decoder_parameters.0"].to(device),
            sd["decoder_parameters.1"].to(device),
        ]

        return v5tg


def save_chunks():
    import utils
    v5c = utils.chunks_v5_from_raw_int16_nonorm("red.s16le", 64, 512)
    v5c_tg = Tensor(v5c.numpy())
    nn.state.safe_save({"red.s16le":v5c_tg}, "red_v5_chunks.safetensors", {"file": "red.s16le", "prefix": 64, "window": 512, "silero-version": "v5"})

def how_much_to_pad(actual_size, multiple):
    rem = actual_size % multiple
    if rem == 0:
        return 0
    else:
        return multiple - rem

def audio_from_raw_int16_tg(filename, sequence_count, device=None)->Tensor:
    red600 = Tensor(pathlib.Path(filename), dtype=dtypes.int16, device=device).float() / (2**15)
    pad = how_much_to_pad(red600.numel(), sequence_count)
    return red600.pad((0, pad), mode="constant").reshape(-1, sequence_count)

def chunks_v5_from_raw_int16_tg(path, prefix, window, device=None)->Tensor:
    red600 = audio_from_raw_int16_tg(path, window, device)
    return red600.flatten().pad((prefix, 0), mode="constant").unfold(0, prefix+window, window)

def foo():
    # v5 = Silero_Vad_5.fromjit("silero-vad-models/v5/silero_vad.jit")
    v5c_tg = nn.state.safe_load("red_v5_chunks.safetensors")["red.s16le"]
    v5_sd_tg = nn.state.safe_load("v5.safetensors")

    forward_basis_buffer = v5_sd_tg["forward_basis_buffer"]

    print(v5c_tg.shape)

    lstm = LSTM()
    nn.state.load_state_dict({"lstm":lstm}, v5_sd_tg)

    tg_stft_out = stft(v5c_tg, forward_basis_buffer)
    print("stft done")
    # np.allclose(torch_stft_out.numpy(), tg_stft_out).numpy()

    tg_encoder_out=encoder(tg_stft_out, [[Tensor(wb.detach().numpy()) for wb in e.parameters()] for e in v5.encoder])
    print("encoder done")
    tg_encoder_out_t = tg_encoder_out.permute(0, 2, 1)
    bs, seq, feat = tg_encoder_out_t.shape
    tg_lstm_minibatched = tg_encoder_out_t.reshape(1, -1, 128)
    # tg_lstm_minibatched = tg_lstm_minibatched[0:1, bbb*0:bbb*1, ...]
    tg_hc = None
    i = 0
    bbb = 75
    output = []
    import tqdm
    pbar = tqdm.tqdm(seq)
    while i*bbb < tg_lstm_minibatched.shape[1]:
        tg_lstm_out, tg_hc = lstm(tg_lstm_minibatched[0:1, bbb*i:bbb*(i+1), ...].contiguous(), tg_hc)
        output.append(tg_lstm_out)
        i+=1
        pbar.update(1*bbb)
    pbar.close()
    print("lstm done")
    tg_lstm_out = Tensor.cat(*output, dim=1).reshape(bs, seq, -1)
    # tg_lstm_out = tg_lstm_out.reshape(bbb, seq, -1)
    tg_lstm_out_t = tg_lstm_out.permute(0, 2, 1)

    def decoder(x:Tensor, decoder):
        weight, bias = list(Tensor(w.detach().numpy()) for w in decoder.decoder.parameters())
        x = x.relu()
        x = x.conv2d(weight=weight, bias=bias)
        x = x.sigmoid()
        return x

    tg_decoder_out = decoder(tg_lstm_out_t, v5.decoder).realize()
    print("decoder done")
    tg_probs = tg_decoder_out.squeeze(1).mean(axis=1).unsqueeze(1)
    print(tg_probs.numpy().flatten())

    # with torch.inference_mode():
    #     torch_stft_out = v5.stft(v5c)
    # torch_stft_out

if __name__ == '__main__':
    # v5 = Silero_Vad_5.fromjit("silero-vad-models/v5/silero_vad.jit")
    # v5t = SileroV5(v5)
    # nn.state.safe_save(nn.state.get_state_dict(v5t), "v5.safetensors")
    # save_chunks()
    # foo()
    with Timing("init model "):
        v5_sd_tg = nn.state.safe_load("v5.safetensors")
        v5t = SileroV5.from_sd(v5_sd_tg, Device.DEFAULT)

    with Timing("load chunks "):
        v5c = chunks_v5_from_raw_int16_tg("red.s16le", 64, 512, Device.DEFAULT)

    BS = getenv("BS", 32)

    def decoder(tg_lstm_out_t:Tensor)->Tensor:
        return tg_lstm_out_t.relu().conv2d(weight=v5t.decoder_parameters[0], bias=v5t.decoder_parameters[1]).sigmoid()

    @TinyJit
    def model(batch:Tensor, tg_h, tg_c):
        tg_stft_out = stft(batch, v5t.forward_basis_buffer)
        tg_encoder_out = encoder(tg_stft_out, v5t.encoder_parameters)
        tg_encoder_out_t = tg_encoder_out.permute(0, 2, 1)
        bs, seq, feat = tg_encoder_out_t.shape
        tg_lstm_minibatched = tg_encoder_out_t.reshape(1, -1, 128)

        tg_lstm_out, tg_hc = v5t.lstm(tg_lstm_minibatched, (tg_h, tg_c))
        tg_lstm_out = tg_lstm_out.reshape(bs, seq, -1)
        # tg_lstm_out = tg_lstm_out.reshape(bbb, seq, -1)
        tg_lstm_out_t = tg_lstm_out.permute(0, 2, 1)
        tg_decoder_out = decoder(tg_lstm_out_t)
        results = tg_decoder_out.squeeze(1).mean(axis=1).flatten()
        return results, tg_hc


    with Timing("red600 "):
        i = 0
        tg_hc = (Tensor.zeros(1, v5t.lstm.cell.weight_hh.size(1), dtype=v5c.dtype, device=v5c.device), Tensor.zeros(1, v5t.lstm.cell.weight_hh.size(1), dtype=v5c.dtype, device=v5c.device))
        pbar = tqdm(total=v5c.shape[0])
        vi = Variable("i", 0, int((math.ceil(v5c.shape[0] / BS) * BS)) - 1)
        probs = []
        while i*BS < v5c.shape[0]:
            vib = vi.bind(i*BS)
            batch = v5c[vib:vib+BS, ...]
            results, tg_hc_ = model(batch, tg_hc[0], tg_hc[1])
            tg_hc[0].assign(tg_hc_[0])
            tg_hc[1].assign(tg_hc_[1])
            probs.append(results.numpy())
            i+=1
            pbar.update(BS)
        with open("probs_tinygrad.txt", "w") as f:
            np.savetxt(f, np.array(probs).flatten(), "%.06f")





# TODO(irwin):
# - all weights to safetensors
# - run whole model end to end, from samples to probabilities
# - verify outputs with torch
# - test CPU
# - speed
# - export?