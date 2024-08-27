# incremental bulky dump
# incremental load
# quic server
# quic client
import torch
import struct
import time
import json
import os

def stream_dump(params_dict, filename, shs_degree=1):
    '''
    name: fromF, toF, xyz, feature, s,   r,    o
    data: I    , I,   fff, fffx4,   fff, ffff, f
    '''
    FORMAT = {
        'start_frame': 'I',
        'end_frame': 'I',
        'xyz': 'fff',
        'f_dc': 'fff',
        'f_rest': 'fff' * ((shs_degree + 1) ** 2 -1),
        'scaling': 'fff',
        'rotation': 'ffff',
        'opacity': 'f'
    }
    ENDIAN = "!"
    N = params_dict['start_frame'].shape[0]
    assert all([p.shape[0] == N for _, p in params_dict.items()]), "Batch size mismatch"
    for k, v in params_dict.items():
        params_dict[k] = v.cpu()

    fmt = f"{ENDIAN}{''.join(FORMAT.values())}"
    print(f"Format: {fmt}, total bytes: {struct.calcsize(fmt)}")
#     print(f'''shape of start_frame: {params_dict['start_frame'].shape}
# shape of end_frame: {params_dict['end_frame'].shape}
# shape of xyz: {params_dict['xyz'].shape}
# shape of f_dc: {params_dict['f_dc'].flatten(1).shape}
# shape of f_rest: {params_dict['f_rest'].flatten(1).shape}
# shape of scaling: {params_dict['scaling'].shape}
# shape of rotation: {params_dict['rotation'].shape}
# shape of opacity: {params_dict['opacity'].shape}
#     ''')
    dir = os.path.dirname(filename)
    with open(os.path.join(dir, 'format.json'), 'w') as f:
        FORMAT['ENDIAN'] = ENDIAN
        json.dump(FORMAT, f, indent=4)

    time_start = time.time()
    values = []
    for i in range(N):
        v = (int(params_dict['start_frame'][i].item()),
            int(params_dict['end_frame'][i].item()),
            *params_dict['xyz'][i].tolist(),
            *params_dict['f_dc'].flatten(1)[i].tolist(),
            *params_dict['f_rest'].flatten(1)[i].tolist(),
            *params_dict['scaling'][i].tolist(),
            *params_dict['rotation'][i].tolist(),
            params_dict['opacity'][i].item()
        )
        values.append(struct.pack(fmt, *v))
    with open(filename, 'ab') as f:
        f.writelines(values)
    time_end = time.time()

    print(f"Dumped {N} gaussians in {time_end - time_start} seconds")

def stream_load(fmtjson, filename):
    with open(fmtjson, 'r') as f:
        FORMAT = json.load(f)
    ENDIAN = FORMAT.pop('ENDIAN')
    fmt = f"{ENDIAN}{''.join(FORMAT.values())}"
    print(f"Format: {fmt}, total bytes: {struct.calcsize(fmt)}")
    with open(filename, 'rb') as f:
        data = f.read()
    N = len(data) // struct.calcsize(fmt)
    print(f"Loading {N} gaussians")
    unpacked = []
    for i in range(N):
        unpacked.append(
            struct.unpack(fmt,
                          data[i * struct.calcsize(fmt): (i+1) * struct.calcsize(fmt)]))
    return unpacked