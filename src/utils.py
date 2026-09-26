import os
import hashlib
from PIL import Image

def compute_sha256(filepath_or_bytes):
    h = hashlib.sha256()
    if isinstance(filepath_or_bytes, (str, bytes, os.PathLike)) and os.path.exists(str(filepath_or_bytes)):
        with open(filepath_or_bytes, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    elif isinstance(filepath_or_bytes, bytes):
        h.update(filepath_or_bytes)
    return h.hexdigest()

def compute_dhash(img, hash_size=8):
    """Computes difference hash (dHash) from a PIL Image."""
    # Resize to (hash_size + 1, hash_size) grayscale
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    # Compare adjacent pixels
    difference = []
    width = hash_size + 1
    for row in range(hash_size):
        row_start = row * width
        for col in range(hash_size):
            pixel_left = pixels[row_start + col]
            pixel_right = pixels[row_start + col + 1]
            difference.append(pixel_left > pixel_right)
    # Convert binary array to hex string
    decimal_val = 0
    hex_string = []
    for index, val in enumerate(difference):
        if val:
            decimal_val += 2 ** (index % 4)
        if (index % 4) == 3:
            hex_string.append(hex(decimal_val)[2:])
            decimal_val = 0
    return "".join(hex_string)

def hamming_distance(h1, h2):
    """Computes Hamming distance between two hex hashes."""
    if len(h1) != len(h2):
        return 999
    n1 = int(h1, 16)
    n2 = int(h2, 16)
    x = n1 ^ n2
    return bin(x).count("1")
