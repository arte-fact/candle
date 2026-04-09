#!/usr/bin/env python3
"""Print architecture and key metadata from GGUF files."""
import struct, sys, os

def read_gguf_meta(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            print(f"  Not a GGUF file (magic: {magic})")
            return
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_metadata = struct.unpack('<Q', f.read(8))[0]
        print(f"  GGUF v{version}, {n_tensors} tensors, {n_metadata} metadata")
        for i in range(min(n_metadata, 30)):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            vtype = struct.unpack('<I', f.read(4))[0]
            if vtype == 8:  # string
                slen = struct.unpack('<Q', f.read(8))[0]
                val = f.read(slen).decode('utf-8')
                print(f"  {key} = \"{val}\"")
            elif vtype == 4:  # uint32
                val = struct.unpack('<I', f.read(4))[0]
                print(f"  {key} = {val}")
            elif vtype == 5:  # int32
                val = struct.unpack('<i', f.read(4))[0]
                print(f"  {key} = {val}")
            elif vtype == 6:  # float32
                val = struct.unpack('<f', f.read(4))[0]
                print(f"  {key} = {val:.4f}")
            elif vtype == 7:  # bool
                val = struct.unpack('B', f.read(1))[0]
                print(f"  {key} = {bool(val)}")
            elif vtype == 10:  # uint64
                val = struct.unpack('<Q', f.read(8))[0]
                print(f"  {key} = {val}")
            else:
                print(f"  {key} (type={vtype}) -- skipping rest")
                break

models_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/artefact/models"
for f in sorted(os.listdir(models_dir)):
    if f.endswith('.gguf') and not f.endswith('.gguf.1') and not f.endswith('.gguf.2'):
        print(f"\n=== {f} ===")
        read_gguf_meta(os.path.join(models_dir, f))
