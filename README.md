# ⚡ InSwapper Pixel Boost (256px Interleaved Grid Sampling)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![ONNX](https://img.shields.io/badge/ONNX_Runtime-InsightFace-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <strong>Break past the native 128x128 resolution bottleneck of InsightFace InSwapper without neural hallucination or smoothing. Interleaved 4-phase spatial decimation and reconstruction for high-fidelity 256px output.</strong>
</p>

</div>

---

## 🔍 The Problem with 128px InSwapper

InsightFace's `inswapper_128.onnx` is trained strictly on a **128x128 pixel** coordinate space. When scaling up:
- **Bicubic / Bilinear downscale & upscale**: Results in soft, blurry faces and lost skin texture.
- **GFPGAN / CodeFormer restorers**: Attempt to regenerate lost details, but frequently introduce AI-hallucinated teeth, plastic skin artifacts, and alter the subject's distinct identity.

---

## 💡 The Interleaved Grid Sampling Approach

Instead of downsampling the image with interpolation filters, **Pixel Boost** decimates a 256x256 facial crop into **4 interleaved sub-grids** of 128x128 by taking every second pixel in both horizontal and vertical axes:

```text
256x256 Crop
┌───┬───┬───┬───┐
│A 0│B 1│A 0│B 1│    Decimate (2x2 stride)
├───┼───┼───┼───┤  ─────────────────────────►  4 Sub-tiles of 128x128:
│C 2│D 3│C 2│D 3│                              Tile 0: (even rows, even cols)
├───┼───┼───┼───┤                              Tile 1: (even rows, odd cols)
│A 0│B 1│A 0│B 1│                              Tile 2: (odd rows, even cols)
├───┼───┼───┼───┤                              Tile 3: (odd rows, odd cols)
│C 2│D 3│C 2│D 3│
└───┴───┴───┴───┘
```

1. Each of the 4 sub-tiles is passed through `inswapper_128.onnx`.
2. Each tile retains distinct high-frequency phase and spatial edge information.
3. The processed tiles are interleaved back into the full 256x256 matrix.
4. **Result:** Crisp, detailed face swap with native pore-level textures and exact identity preservation.

---

## ⚡ Performance

- **Pure Vectorized NumPy:** Splitting and merging take **< 0.8 ms** per frame on modern CPU.
- **Zero Loss:** Lossless roundtrip verified (100% exact parity on unswapped grids).
- **Compatible with Real-Time Pipelines:** Easily integrates into webcam loopback streamers, ComfyUI custom nodes, and offline video processors.

---

## 🚀 Installation & Usage

```bash
git clone https://github.com/ProgramaGrueso/inswapper-pixel-boost.git
cd inswapper-pixel-boost
pip install -e .
```

### Python API

```python
import cv2
from pixel_boost import split_interleaved_tiles, merge_interleaved_tiles

# 1. Prepare 256x256 aligned face crop
crop_256 = cv2.warpAffine(frame, affine_matrix, (256, 256))

# 2. Split into 4 interleaved 128x128 tiles
tiles = split_interleaved_tiles(crop_256, target_size=128) # Shape: (4, 128, 128, 3)

# 3. Run InSwapper on each tile
swapped_tiles = [run_inswapper_128(tile) for tile in tiles]

# 4. Reconstruct to full 256x256
swapped_256 = merge_interleaved_tiles(swapped_tiles, crop_size=256, target_size=128)
```

### Run Benchmark

```bash
python cli.py --size 256 --sub-size 128 --iterations 1000
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
