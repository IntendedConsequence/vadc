from pathlib import Path
import numpy as np
import struct

import torch

def serialize_numpy_array(arr):
    try:
        arr = arr.numpy()
    except:
        pass
    # Ensure the array is of the correct type
    arr = arr.astype(np.float32)

    # Prepare dimensions data
    dims = list(arr.shape)
    ndims = arr.ndim

    # Serialize dimensions and number of dimensions
    serialized = struct.pack('i', ndims)
    if ndims > 0:
        serialized += struct.pack(f'{ndims}i', *dims)
    serialized += struct.pack('i', arr.size)
    serialized += struct.pack('i', arr.nbytes)

    # Serialize data
    serialized += arr.tobytes()

    return serialized

def serialize_multiple_arrays(arrays):
    array_dict = {}
    for name, array in arrays.items():
        try:
            array = array.numpy()
        except:
            pass
        if array.ndim == 0:
            print(f"Skipping 0-dim array {name}")
        elif array.dtype != np.float32:
            print(f"Skipping non-float32 (dtype {array.dtype}) array {name}")
        else:
            array_dict[name] = array
    version = 1
    serialized = struct.pack('ii', version, len(array_dict))
    for arr_name in array_dict:
        encoded = arr_name.encode('utf8')
        serialized += struct.pack('i', len(encoded))
        serialized += encoded
    for arr in array_dict.values():
        serialized += serialize_numpy_array(arr)

    return serialized

def state_dict_from_bytes(serialized_data):
    version, num_arrays = struct.unpack('ii', serialized_data[:8])
    assert version == 1, f"Unsupported version {version}"
    offset = 8
    names = []
    state_dict = {}
    for i in range(num_arrays):
        name_len = struct.unpack('i', serialized_data[offset:offset+4])[0]
        offset += 4
        name = serialized_data[offset:offset+name_len].decode('utf8')
        offset += name_len
        names.append(name)

    for name in names:
        # print(name)
        ndim = struct.unpack('i', serialized_data[offset:offset+4])[0]
        offset += 4
        # print(ndim)
        if ndim > 0:
            dims = struct.unpack(f'{ndim}i', serialized_data[offset:offset+4*ndim])
            offset += 4 * ndim
        else:
            dims = []
        # print(dims)
        size = struct.unpack('i', serialized_data[offset:offset+4])[0]
        offset += 4
        nbytes = struct.unpack('i', serialized_data[offset:offset+4])[0]
        offset += 4
        array = np.frombuffer(serialized_data[offset:offset+nbytes], dtype=np.float32).reshape(dims)
        offset += nbytes
        state_dict[name] = array
    return state_dict

# Example usage
# arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
# serialized_data = serialize_multiple_arrays(state_dict)
# Path('test.testtensor').write_bytes(serialized_data)

def prepare_lstm_weights_and_biases_for_c(state_dict):
    ih_hh_l0 = np.concatenate([state_dict['_model1.lstm.weight_ih_l0'], state_dict['_model1.lstm.weight_hh_l0']], -1)
    ih_hh_l1 = np.concatenate([state_dict['_model1.lstm.weight_ih_l1'], state_dict['_model1.lstm.weight_hh_l1']], -1)

    weights = np.stack([ih_hh_l0, ih_hh_l1])
    # NOTE(irwin): we add biases since in vanilla LSTM they are fused, but in torch they are separate for CUDA compatibility
    biases_l0 = state_dict['_model1.lstm.bias_ih_l0'] + state_dict['_model1.lstm.bias_hh_l0']
    biases_l1 = state_dict['_model1.lstm.bias_ih_l1'] + state_dict['_model1.lstm.bias_hh_l1']
    biases = np.stack([biases_l0, biases_l1])
    lstm_weights_dict = {
        'weights': weights,
        'biases': biases,
    }

    return lstm_weights_dict

def serialize_lstm_weights_and_biases_for_c(state_dict):
    lstm_dict = prepare_lstm_weights_and_biases_for_c(state_dict)
    lstm_bytes = serialize_multiple_arrays(lstm_dict)
    Path('lstm_silero_3.1_16k_for_c.testtensor').write_bytes(lstm_bytes)

def transformer_l1_key_map():
    map = {}
    i = 0
    map['dw_conv_weights'] = '_model1.first_layer.0.dw_conv.0.weight'
    map['dw_conv_biases'] = '_model1.first_layer.0.dw_conv.0.bias'
    map['pw_conv_weights'] = '_model1.first_layer.0.pw_conv.0.weight'
    map['pw_conv_biases'] = '_model1.first_layer.0.pw_conv.0.bias'
    map['proj_weights'] = '_model1.first_layer.0.proj.weight'
    map['proj_biases'] = '_model1.first_layer.0.proj.bias'

    map['attention_weights'] = f'_model1.encoder.{i}.attention.QKV.weight'
    map['attention_biases'] = f'_model1.encoder.{i}.attention.QKV.bias'
    map['attention_proj_weights'] = f'_model1.encoder.{i}.attention.out_proj.weight'
    map['attention_proj_biases'] = f'_model1.encoder.{i}.attention.out_proj.bias'

    map['norm1_weights'] = f'_model1.encoder.{i}.norm1.weight'
    map['norm1_biases'] = f'_model1.encoder.{i}.norm1.bias'
    map['linear1_weights'] = f'_model1.encoder.{i}.linear1.weight'
    map['linear1_biases'] = f'_model1.encoder.{i}.linear1.bias'
    map['linear2_weights'] = f'_model1.encoder.{i}.linear2.weight'
    map['linear2_biases'] = f'_model1.encoder.{i}.linear2.bias'
    map['norm2_weights'] = f'_model1.encoder.{i}.norm2.weight'
    map['norm2_biases'] = f'_model1.encoder.{i}.norm2.bias'

    i += 1
    map['conv_weights'] = f'_model1.encoder.{i}.weight'
    map['conv_biases'] = f'_model1.encoder.{i}.bias'

    i += 1
    map['batch_norm_weights'] = f'_model1.encoder.{i}.weight'
    map['batch_norm_biases'] = f'_model1.encoder.{i}.bias'
    map['batch_norm_running_mean'] = f'_model1.encoder.{i}.running_mean'
    map['batch_norm_running_var'] = f'_model1.encoder.{i}.running_var'

    return map

def transformer_l2_key_map(i):
    map = {}

    map['dw_conv_weights'] = f'_model1.encoder.{i}.0.dw_conv.0.weight'
    map['dw_conv_biases'] = f'_model1.encoder.{i}.0.dw_conv.0.bias'
    map['pw_conv_weights'] = f'_model1.encoder.{i}.0.pw_conv.0.weight'
    map['pw_conv_biases'] = f'_model1.encoder.{i}.0.pw_conv.0.bias'
    map['proj_weights'] = f'_model1.encoder.{i}.0.proj.weight'
    map['proj_biases'] = f'_model1.encoder.{i}.0.proj.bias'

    i += 1
    map['attention_weights'] = f'_model1.encoder.{i}.attention.QKV.weight'
    map['attention_biases'] = f'_model1.encoder.{i}.attention.QKV.bias'
    map['attention_proj_weights'] = f'_model1.encoder.{i}.attention.out_proj.weight'
    map['attention_proj_biases'] = f'_model1.encoder.{i}.attention.out_proj.bias'

    map['norm1_weights'] = f'_model1.encoder.{i}.norm1.weight'
    map['norm1_biases'] = f'_model1.encoder.{i}.norm1.bias'
    map['linear1_weights'] = f'_model1.encoder.{i}.linear1.weight'
    map['linear1_biases'] = f'_model1.encoder.{i}.linear1.bias'
    map['linear2_weights'] = f'_model1.encoder.{i}.linear2.weight'
    map['linear2_biases'] = f'_model1.encoder.{i}.linear2.bias'
    map['norm2_weights'] = f'_model1.encoder.{i}.norm2.weight'
    map['norm2_biases'] = f'_model1.encoder.{i}.norm2.bias'

    i += 1
    map['conv_weights'] = f'_model1.encoder.{i}.weight'
    map['conv_biases'] = f'_model1.encoder.{i}.bias'

    i += 1
    map['batch_norm_weights'] = f'_model1.encoder.{i}.weight'
    map['batch_norm_biases'] = f'_model1.encoder.{i}.bias'
    map['batch_norm_running_mean'] = f'_model1.encoder.{i}.running_mean'
    map['batch_norm_running_var'] = f'_model1.encoder.{i}.running_var'

    return map

def transformer_l3_key_map(i):
    map = transformer_l2_key_map(i)
    del map["proj_weights"]
    del map["proj_biases"]

    return map

def prepare_silero_v31_weights(state_dict):
    weight_dict = {}

    weight_dict['forward_basis_buffer'] = state_dict['_model1.feature_extractor.forward_basis_buffer']

    l1_key_map = transformer_l1_key_map()
    l2_key_map = transformer_l2_key_map(4)
    l3_key_map = transformer_l3_key_map(9)
    l4_key_map = transformer_l2_key_map(14)

    for key in l1_key_map:
        weight_dict[f"transformer_l1.{key}"] = state_dict[l1_key_map[key]]

    for key in l2_key_map:
        weight_dict[f"transformer_l2.{key}"] = state_dict[l2_key_map[key]]

    for key in l3_key_map:
        weight_dict[f"transformer_l3.{key}"] = state_dict[l3_key_map[key]]

    for key in l4_key_map:
        weight_dict[f"transformer_l4.{key}"] = state_dict[l4_key_map[key]]

    lstm_weights = prepare_lstm_weights_and_biases_for_c(state_dict)
    weight_dict.update(lstm_weights)

    weight_dict['decoder_weights'] = state_dict['_model1.decoder.1.weight']
    weight_dict['decoder_biases'] = state_dict['_model1.decoder.1.bias']

    return weight_dict


def serialize_silero_v31_weights_16k():
    jit_model = torch.jit.load(r"silero-vad-models\v3.1\silero_vad.jit")
    jit_model.eval()

    sd = prepare_silero_v31_weights(jit_model.state_dict())
    ser = serialize_multiple_arrays(sd)
    print(len(ser))
    Path('testdata/silero_v31_16k.testtensor').write_bytes(ser)

def how_much_to_pad(actual_size, multiple):
    rem = actual_size % multiple
    if rem == 0:
        return 0
    else:
        return multiple - rem

def audio_from_raw_int16_unpadded(filename):
    audio_data = torch.from_numpy(np.fromfile(filename, dtype=np.int16)).float()
    audio_data /= 32768.0
    return audio_data

def audio_from_raw_int16(filename, sequence_count):
    audio_data = torch.from_numpy(np.fromfile(filename, dtype=np.int16)).float()
    audio_data /= 32768.0

    size = audio_data.size(0)
    pad = how_much_to_pad(size, sequence_count)

    audio_data_padded = torch.nn.functional.pad(audio_data, (0, pad), mode="constant")
    return audio_data_padded.reshape(-1, sequence_count)

def normalized_audio_from_raw_int16(filename, sequence_count, normalization_window=None):
    if normalization_window is None:
        normalization_window = sequence_count

    audio_data = torch.from_numpy(np.fromfile(filename, dtype=np.int16)).float()
    # audio_data /= audio_data.abs().max()

    size = audio_data.size(0)
    pad = how_much_to_pad(size, normalization_window)

    audio_data_padded = torch.nn.functional.pad(audio_data, (0, pad), mode="constant")
    audio_data_chunked = audio_data_padded.reshape(-1, normalization_window)
    local_abs_maximums = audio_data_chunked.abs().max(axis=1, keepdim=True)[0]
    audio_data_normalized = audio_data_chunked / local_abs_maximums

    audio_data_ = audio_data_normalized.reshape(-1)[:size]
    pad2 = how_much_to_pad(size, sequence_count)
    padded2 = torch.nn.functional.pad(audio_data_, (0, pad2), mode="constant")

    return padded2.reshape(-1, sequence_count)

def chunks_v5_from_raw_int16(path, prefix, window):
    cont = normalized_audio_from_raw_int16(path, window)
    return torch.nn.functional.pad(cont.flatten(), (prefix, 0), mode='constant', value=0.0).unfold(0, window+prefix, window)

def chunks_v5_from_raw_int16_nonorm(path, prefix, window):
    cont = audio_from_raw_int16(path, window)
    return torch.nn.functional.pad(cont.flatten(), (prefix, 0), mode='constant', value=0.0).unfold(0, window+prefix, window)