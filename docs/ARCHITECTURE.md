# VisionKit Architecture

## Overview

VisionKit is a lightweight Python library for image processing and basic computer vision tasks. It is designed around simplicity, composability, and minimal dependencies.

## Module Structure

```
src/visionkit/
├── __init__.py      # Public API exports (VisionKit, VisionKitConfig)
├── core.py          # Main VisionKit class with all CV operations
├── config.py        # Pydantic-based configuration from environment
└── utils.py         # Image conversion helpers and color space math
```

## Core Components

### VisionKit (core.py)

The central class that exposes all image operations. It is stateless — each method takes an image and returns a result without mutating the input.

**Key methods:**

| Method             | Description                                    | Algorithm               |
|--------------------|------------------------------------------------|-------------------------|
| `load_image`       | Load from file path or URL                     | PIL.Image.open          |
| `resize`           | Resize with aspect ratio preservation          | Lanczos resampling      |
| `detect_edges`     | Edge detection                                 | Sobel operator (NumPy)  |
| `detect_colors`    | Dominant color extraction                      | K-means clustering      |
| `compute_histogram`| Per-channel color histogram                    | NumPy bincount          |
| `compare_images`   | Structural similarity between two images       | Mean absolute error     |
| `extract_metadata` | EXIF and basic image properties                | PIL EXIF parser         |
| `apply_filter`     | Apply blur, sharpen, grayscale, sepia, etc.    | PIL filters + NumPy     |

### VisionKitConfig (config.py)

Uses Pydantic `BaseModel` for validated configuration. Reads from environment variables with sensible defaults:

- `LOG_LEVEL` — Controls logging verbosity (default: INFO)
- `MAX_IMAGE_SIZE` — Guards against loading excessively large images (default: 4096)

### Utils (utils.py)

Pure-function helpers with no side effects:

- **Image format conversion** — PIL to NumPy and back
- **Color space conversion** — RGB to grayscale (BT.601 luma), RGB to HSV, HSV to RGB
- **Validation** — Dimension bounds checking
- **Aspect ratio math** — Compute target dimensions preserving ratio

## Design Principles

1. **No heavy dependencies** — Only Pillow, NumPy, and Pydantic. No OpenCV, no PyTorch.
2. **Real math** — Edge detection uses actual Sobel convolution; color clustering uses real k-means iterations; similarity uses MAE over pixel arrays.
3. **Stateless operations** — Every method on `VisionKit` is a pure transform. Input images are never mutated.
4. **Fail fast** — Dimension validation happens at load time. Unknown filters raise `ValueError` immediately.
5. **Configurable via environment** — Pydantic config reads from env vars, making it easy to customize in containers and CI.

## Data Flow

```
User Code
    │
    ▼
VisionKit.load_image(path)
    │  → validates dimensions via config.max_image_size
    ▼
PIL.Image (RGB)
    │
    ▼
VisionKit.<operation>(image, ...)
    │  → converts to NumPy via utils.pil_to_numpy()
    │  → performs math (convolution, clustering, etc.)
    │  → converts back via utils.numpy_to_pil()
    ▼
Result (PIL.Image, dict, float, or list)
```

## Testing Strategy

All tests use **synthetic images** created at test time with PIL and NumPy:

- Solid color images (for histogram and color detection tests)
- Gradient images (for edge detection)
- Checkerboard patterns (for edge detection validation)

No external test fixtures or downloaded images are required.
