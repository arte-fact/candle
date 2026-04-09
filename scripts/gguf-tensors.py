#!/usr/bin/env python3
"""List tensor names from a GGUF file."""
import struct, sys

def read_string(f):
    slen = struct.unpack('<Q', f.read(8))[0]
    return f.read(slen).decode('utf-8')

path = sys.argv[1]
with open(path, 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_metadata = struct.unpack('<Q', f.read(8))[0]

    # Skip metadata
    for i in range(n_metadata):
        key_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8')
        vtype = struct.unpack('<I', f.read(4))[0]
        if vtype == 8:  # string
            slen = struct.unpack('<Q', f.read(8))[0]
            f.read(slen)
        elif vtype in (4, 5):  # uint32/int32
            f.read(4)
        elif vtype == 6:  # float32
            f.read(4)
        elif vtype == 7:  # bool
            f.read(1)
        elif vtype == 10:  # uint64
            f.read(8)
        elif vtype == 9:  # array
            atype = struct.unpack('<I', f.read(4))[0]
            alen = struct.unpack('<Q', f.read(8))[0]
            for _ in range(alen):
                if atype == 8:
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.read(slen)
                elif atype in (4, 5):
                    f.read(4)
                elif atype == 6:
                    f.read(4)
                elif atype == 7:
                    f.read(1)
                elif atype == 10:
                    f.read(8)
                else:
                    break
        else:
            break

    # Read tensor info
    print(f"{n_tensors} tensors:")
    # Only show first block + output to identify naming pattern
    for i in range(n_tensors):
        name = read_string(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        if 'blk.0.' in name or 'output' in name or 'token' in name or 'norm' in name:
            print(f"  {name} [{', '.join(str(d) for d in dims)}] type={dtype}")
