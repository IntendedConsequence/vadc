import numpy as np
import struct

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
