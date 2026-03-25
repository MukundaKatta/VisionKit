"""Core VisionKit class — image processing, feature detection, and analysis."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageFilter

from visionkit.config import VisionKitConfig
from visionkit.utils import (
    compute_aspect_ratio,
    numpy_to_pil,
    pil_to_numpy,
    rgb_to_grayscale,
    validate_image_dimensions,
)

logger = logging.getLogger("visionkit")


class VisionKit:
    """Lightweight computer vision toolkit built on Pillow and NumPy."""

    def __init__(self, config: VisionKitConfig | None = None) -> None:
        self.config = config or VisionKitConfig.from_env()
        self.config.configure_logging()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_image(self, path_or_url: str) -> Image.Image:
        """Load an image from a local file path or a URL.

        Returns an RGB PIL Image.
        """
        if path_or_url.startswith(("http://", "https://")):
            logger.info("Downloading image from %s", path_or_url)
            with urlopen(path_or_url) as resp:  # noqa: S310
                data = resp.read()
            img = Image.open(io.BytesIO(data))
        else:
            logger.info("Loading image from %s", path_or_url)
            img = Image.open(Path(path_or_url))

        img = img.convert("RGB")
        validate_image_dimensions(
            img.width, img.height, self.config.max_image_size
        )
        return img

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def resize(
        self,
        image: Image.Image,
        width: int | None = None,
        height: int | None = None,
    ) -> Image.Image:
        """Resize an image, preserving aspect ratio when only one dim given."""
        new_w, new_h = compute_aspect_ratio(
            image.width, image.height, width, height
        )
        validate_image_dimensions(new_w, new_h, self.config.max_image_size)
        logger.info("Resizing from %dx%d to %dx%d", image.width, image.height, new_w, new_h)
        return image.resize((new_w, new_h), Image.LANCZOS)

    # ------------------------------------------------------------------
    # Edge detection (Sobel)
    # ------------------------------------------------------------------

    def detect_edges(
        self, image: Image.Image, threshold: int = 50
    ) -> Image.Image:
        """Detect edges using the Sobel operator.

        Converts to grayscale, applies horizontal and vertical Sobel kernels,
        computes gradient magnitude, and thresholds the result.
        """
        gray = rgb_to_grayscale(pil_to_numpy(image)).astype(np.float64)

        # Sobel kernels
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

        h, w = gray.shape
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)

        # Convolve (skip border pixels)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                patch = gray[i - 1 : i + 2, j - 1 : j + 2]
                gx[i, j] = np.sum(patch * kx)
                gy[i, j] = np.sum(patch * ky)

        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = (magnitude / magnitude.max() * 255) if magnitude.max() > 0 else magnitude
        edges = np.where(magnitude >= threshold, 255, 0).astype(np.uint8)

        logger.info("Edge detection complete (threshold=%d)", threshold)
        return numpy_to_pil(edges)

    # ------------------------------------------------------------------
    # Dominant color extraction (mini k-means)
    # ------------------------------------------------------------------

    def detect_colors(
        self, image: Image.Image, n_colors: int = 5, max_iterations: int = 20
    ) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using a simplified k-means algorithm.

        Downsamples the image first for speed, then clusters pixel colors.
        Returns a list of (R, G, B) tuples sorted by cluster size descending.
        """
        # Downsample for speed
        thumb = image.copy()
        thumb.thumbnail((100, 100))
        pixels = pil_to_numpy(thumb).reshape(-1, 3).astype(np.float64)

        n = len(pixels)
        n_colors = min(n_colors, n)

        # Initialise centroids from evenly-spaced pixel indices
        indices = np.linspace(0, n - 1, n_colors, dtype=int)
        centroids = pixels[indices].copy()

        labels = np.zeros(n, dtype=int)

        for _ in range(max_iterations):
            # Assign each pixel to nearest centroid
            dists = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # Recompute centroids
            for k in range(n_colors):
                members = pixels[labels == k]
                if len(members) > 0:
                    centroids[k] = members.mean(axis=0)

        # Sort by cluster size (descending)
        counts = np.bincount(labels, minlength=n_colors)
        order = np.argsort(-counts)
        result = [
            (int(centroids[k][0]), int(centroids[k][1]), int(centroids[k][2]))
            for k in order
        ]
        logger.info("Extracted %d dominant colors", len(result))
        return result

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    def compute_histogram(
        self, image: Image.Image
    ) -> Dict[str, List[int]]:
        """Compute per-channel color histograms (256 bins each).

        Returns a dict with keys 'red', 'green', 'blue', each mapping to a
        list of 256 integer counts.
        """
        arr = pil_to_numpy(image)
        hist: Dict[str, List[int]] = {}
        for idx, channel in enumerate(("red", "green", "blue")):
            counts, _ = np.histogram(arr[:, :, idx], bins=256, range=(0, 256))
            hist[channel] = counts.tolist()
        logger.info("Histogram computed for %dx%d image", image.width, image.height)
        return hist

    # ------------------------------------------------------------------
    # Image comparison (structural similarity)
    # ------------------------------------------------------------------

    def compare_images(
        self, img1: Image.Image, img2: Image.Image
    ) -> float:
        """Compute a similarity score between two images (0.0 to 1.0).

        Both images are resized to the same dimensions, converted to
        grayscale, and compared using a simplified structural similarity
        metric (mean absolute error mapped to [0, 1]).
        """
        size = (min(img1.width, img2.width), min(img1.height, img2.height))
        a = rgb_to_grayscale(pil_to_numpy(img1.resize(size, Image.LANCZOS))).astype(np.float64)
        b = rgb_to_grayscale(pil_to_numpy(img2.resize(size, Image.LANCZOS))).astype(np.float64)

        # Mean absolute error normalised to [0, 1]
        mae = np.mean(np.abs(a - b)) / 255.0
        similarity = 1.0 - mae

        logger.info("Image similarity: %.4f", similarity)
        return float(round(similarity, 4))

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def extract_metadata(self, image: Image.Image) -> Dict[str, Any]:
        """Extract basic metadata and EXIF data (if present) from an image."""
        meta: Dict[str, Any] = {
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "num_pixels": image.width * image.height,
        }

        exif_data = image.getexif()
        if exif_data:
            meta["exif"] = {
                tag_id: str(value)
                for tag_id, value in exif_data.items()
            }
        else:
            meta["exif"] = {}

        logger.info("Metadata extracted for %s image", image.mode)
        return meta

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    _PIL_FILTERS = {
        "blur": ImageFilter.GaussianBlur(radius=2),
        "sharpen": ImageFilter.SHARPEN,
        "detail": ImageFilter.DETAIL,
        "contour": ImageFilter.CONTOUR,
        "emboss": ImageFilter.EMBOSS,
        "smooth": ImageFilter.SMOOTH,
    }

    def apply_filter(
        self, image: Image.Image, filter_type: str
    ) -> Image.Image:
        """Apply a named filter to an image.

        Supported filter_type values:
            blur, sharpen, detail, contour, emboss, smooth, grayscale, sepia
        """
        ft = filter_type.lower().strip()

        if ft == "grayscale":
            gray = rgb_to_grayscale(pil_to_numpy(image))
            rgb = np.stack([gray, gray, gray], axis=-1)
            return numpy_to_pil(rgb)

        if ft == "sepia":
            arr = pil_to_numpy(image).astype(np.float64)
            sepia_matrix = np.array(
                [
                    [0.393, 0.769, 0.189],
                    [0.349, 0.686, 0.168],
                    [0.272, 0.534, 0.131],
                ]
            )
            sepia = arr @ sepia_matrix.T
            sepia = np.clip(sepia, 0, 255).astype(np.uint8)
            return numpy_to_pil(sepia)

        pil_filter = self._PIL_FILTERS.get(ft)
        if pil_filter is None:
            raise ValueError(
                f"Unknown filter '{filter_type}'. "
                f"Supported: {', '.join(list(self._PIL_FILTERS) + ['grayscale', 'sepia'])}"
            )

        logger.info("Applying filter: %s", ft)
        return image.filter(pil_filter)
