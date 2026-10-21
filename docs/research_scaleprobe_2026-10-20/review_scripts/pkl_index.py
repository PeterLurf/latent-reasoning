import sys, struct, pickle, collections
def parse(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'PK\x03\x04', b[:4]
    nlen, xlen = struct.unpack('<HH', b[26:30]); csize = struct.unpack('<I', b[18:22])[0]
    name = b[30:30+nlen].decode(); start = 30+nlen+xlen
    if csize in (0, 0xffffffff):
        end = b.find(b'PK\x07\x08', start)
        if end < 0: end = b.find(b'PK\x03\x04', start)
    else: end = start+csize
    data = b[start:end]
    class St:  # storage stub
        def __init__(s, *a): s.a = a
    class U(pickle.Unpickler):
        def persistent_load(s, pid):
            return ("storage", pid[1], pid[2], pid[4])
        def find_class(s, mod, nm):
            if nm == '_rebuild_tensor_v2':
                return lambda storage, off, size, stride, *a: ('T', storage[1], tuple(size))
            if nm == 'OrderedDict': return collections.OrderedDict
            if mod.startswith("torch") and nm.endswith("Storage"): return nm
            if mod.startswith("torch") : return lambda *a, **k: ("obj", nm, a)
            return super().find_class(mod, nm)
    sd = U(__import__('io').BytesIO(data)).load()
    return name, sd
name, sd = parse(sys.argv[1]); print('entry', name, 'nkeys', len(sd))
tot = collections.Counter(); 
import math
for k, v in sd.items():
    if isinstance(v, tuple) and v[0] == 'T':
        tot[v[1]] += math.prod(v[2])
show = [k for k in sd if not any(f'layers.{i}.' in k for i in range(1, 99))]
for k in show:
    v = sd[k]; print(' ', k, v[1:] if isinstance(v, tuple) else type(v))
print('params by storage type', dict(tot))
