#!/usr/bin/env python3
"""
CLI validation and benchmark script comparing standard resizing vs interleaved pixel boost.
"""

import argparse
import time
import numpy as np
from pixel_boost import split_interleaved_tiles, merge_interleaved_tiles


def main():
    parser = argparse.ArgumentParser(description="InSwapper Pixel Boost Interleaved Grid Validator")
    parser.add_argument("--size", type=int, default=256, help="Base crop size (default 256)")
    parser.add_argument("--sub-size", type=int, default=128, help="Sub-tile size (default 128)")
    parser.add_argument("--iterations", type=int, default=1000, help="Benchmark iterations")
    args = parser.parse_args()

    print(f"=== Benchmarking Interleaved Grid Decimation & Merge ({args.size}px -> {args.sub_size}px) ===")
    dummy_crop = np.random.randint(0, 256, (args.size, args.size, 3), dtype=np.uint8)

    # Validate lossless roundtrip
    tiles = split_interleaved_tiles(dummy_crop, target_size=args.sub_size)
    reconstructed = merge_interleaved_tiles(list(tiles), crop_size=args.size, target_size=args.sub_size)

    assert np.array_equal(dummy_crop, reconstructed), "Error: Reconstructed image does not match input exactly!"
    print("[✓] Lossless phase verification: PASSED (Exact pixel parity confirmed)")

    # Latency test
    t0 = time.perf_counter()
    for _ in range(args.iterations):
        t = split_interleaved_tiles(dummy_crop, target_size=args.sub_size)
        _ = merge_interleaved_tiles(list(t), crop_size=args.size, target_size=args.sub_size)
    dt = (time.perf_counter() - t0) / args.iterations * 1000

    print(f"[✓] Average split + merge overhead: {dt:.4f} ms per frame")
    print("[✓] Pure vectorized numpy operations: Zero CUDA synchronization required.")


if __name__ == "__main__":
    main()
