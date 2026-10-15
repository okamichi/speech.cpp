"""Writes the audio of a dump's .npy file, mono float32 samples such as audio.npy or wav.npy, as a 32-bit float WAVE
file at the sample rate given, which the dump's model takes or makes: 16000 for the recognition and detection models'
audio.npy, 24000 for Qwen3-TTS's wav.npy and 48000 for Irodori-TTS's audio.npy. The samples are written as they are,
so that `speech asr` or a player reads the dump's own audio.

usage: python3 reference/npy_to_wav.py <in.npy> <sample rate> <out.wav>
"""

import ast
import struct
import sys

if len(sys.argv) != 4:
    raise SystemExit(__doc__)
source, rate, target = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(source, "rb") as f:
    if f.read(6) != b"\x93NUMPY":
        raise SystemExit(f"{source} is not a .npy file")
    major = f.read(2)[0]
    length = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
    header = ast.literal_eval(f.read(length).decode("latin1"))
    if header["descr"] != "<f4" or len(header["shape"]) != 1 or header["fortran_order"]:
        raise SystemExit(f"{source} holds {header['descr']} of shape {header['shape']}; give a file of mono float32 samples")
    data = f.read()
fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
with open(target, "wb") as f:
    f.write(b"RIFF" + struct.pack("<I", len(body)) + body)
print(f"{target}: {len(data) // 4} samples, {len(data) / 4 / rate:.2f} s at {rate} Hz")
