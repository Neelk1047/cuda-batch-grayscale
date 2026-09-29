#!/usr/bin/env python3
import argparse
import os
import random

def write_ppm(path, width, height, seed):
    rng = random.Random(seed)
    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        # Deterministic synthetic RGB image.
        row = bytearray(width * 3)
        for y in range(height):
            for x in range(width):
                i = x * 3
                row[i] = (x + seed * 7) % 256
                row[i + 1] = (y * 2 + seed * 11) % 256
                row[i + 2] = (x + y + rng.randrange(32)) % 256
            f.write(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--output", default="images/input")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    for i in range(args.count):
        write_ppm(
            os.path.join(args.output, f"image_{i:04d}.ppm"),
            args.width, args.height, i + 1
        )
    print(f"Generated {args.count} PPM images in {args.output}")

if __name__ == "__main__":
    main()
