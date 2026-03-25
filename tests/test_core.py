"""Tests for VisionKit core functionality using synthetic images."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from visionkit import VisionKit


@pytest.fixture
def vk() -> VisionKit:
    """Return a VisionKit instance with default config."""
    return VisionKit()


@pytest.fixture
def solid_red_image() -> Image.Image:
    """Create a 100x100 solid red image."""
    return Image.new("RGB", (100, 100), color=(255, 0, 0))


@pytest.fixture
def gradient_image() -> Image.Image:
    """Create a 100x100 horizontal gradient image (black to white)."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    for col in range(100):
        val = int(col * 255 / 99)
        arr[:, col, :] = val
    return Image.fromarray(arr)


@pytest.fixture
def checkerboard_image() -> Image.Image:
    """Create a 100x100 black-and-white checkerboard (10px squares)."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            if (i // 10 + j // 10) % 2 == 0:
                arr[i, j, :] = 255
    return Image.fromarray(arr)


# ------------------------------------------------------------------
# Test: image creation and resize
# ------------------------------------------------------------------


class TestResize:
    def test_resize_with_both_dims(
        self, vk: VisionKit, solid_red_image: Image.Image
    ) -> None:
        resized = vk.resize(solid_red_image, width=50, height=50)
        assert resized.size == (50, 50)

    def test_resize_preserves_aspect_ratio_width_only(
        self, vk: VisionKit, solid_red_image: Image.Image
    ) -> None:
        resized = vk.resize(solid_red_image, width=50)
        assert resized.size == (50, 50)  # 100x100 -> 50x50

    def test_resize_preserves_aspect_ratio_rectangular(
        self, vk: VisionKit
    ) -> None:
        img = Image.new("RGB", (200, 100), color=(0, 0, 255))
        resized = vk.resize(img, width=100)
        assert resized.size == (100, 50)

    def test_resize_preserves_content(
        self, vk: VisionKit, solid_red_image: Image.Image
    ) -> None:
        resized = vk.resize(solid_red_image, width=50, height=50)
        arr = np.array(resized)
        # Should still be red
        assert arr[:, :, 0].mean() > 250
        assert arr[:, :, 1].mean() < 5
        assert arr[:, :, 2].mean() < 5


# ------------------------------------------------------------------
# Test: edge detection on synthetic image
# ------------------------------------------------------------------


class TestEdgeDetection:
    def test_edges_on_checkerboard(
        self, vk: VisionKit, checkerboard_image: Image.Image
    ) -> None:
        edges = vk.detect_edges(checkerboard_image, threshold=30)
        assert edges.size == checkerboard_image.size
        arr = np.array(edges)
        # Checkerboard should have visible edges — some white pixels
        assert arr.max() == 255
        # There should be a mix of edge and non-edge pixels
        edge_ratio = np.sum(arr == 255) / arr.size
        assert 0.01 < edge_ratio < 0.99

    def test_edges_on_solid_returns_no_edges(
        self, vk: VisionKit, solid_red_image: Image.Image
    ) -> None:
        edges = vk.detect_edges(solid_red_image, threshold=50)
        arr = np.array(edges)
        # Solid image should have zero edges
        assert arr.max() == 0

    def test_edges_on_gradient(
        self, vk: VisionKit, gradient_image: Image.Image
    ) -> None:
        edges = vk.detect_edges(gradient_image, threshold=10)
        arr = np.array(edges)
        # Gradient has constant horizontal change -> vertical edge lines
        assert arr.max() == 255


# ------------------------------------------------------------------
# Test: histogram computation
# ------------------------------------------------------------------


class TestHistogram:
    def test_histogram_shape(
        self, vk: VisionKit, solid_red_image: Image.Image
    ) -> None:
        hist = vk.compute_histogram(solid_red_image)
        assert set(hist.keys()) == {"red", "green", "blue"}
        for channel in hist.values():
            assert len(channel) == 256

    def test_histogram_solid_red(
        self, vk: VisionKit, solid_red_image: Image.Image
    ) -> None:
        hist = vk.compute_histogram(solid_red_image)
        # All 10000 pixels should be in the 255 bin for red
        assert hist["red"][255] == 100 * 100
        # Green and blue should all be in the 0 bin
        assert hist["green"][0] == 100 * 100
        assert hist["blue"][0] == 100 * 100

    def test_histogram_total_counts(
        self, vk: VisionKit, gradient_image: Image.Image
    ) -> None:
        hist = vk.compute_histogram(gradient_image)
        total_pixels = 100 * 100
        for channel in ("red", "green", "blue"):
            assert sum(hist[channel]) == total_pixels


# ------------------------------------------------------------------
# Test: dominant color detection
# ------------------------------------------------------------------


class TestDetectColors:
    def test_solid_image_dominant_color(
        self, vk: VisionKit, solid_red_image: Image.Image
    ) -> None:
        colors = vk.detect_colors(solid_red_image, n_colors=3)
        assert len(colors) >= 1
        r, g, b = colors[0]
        assert r > 200
        assert g < 50
        assert b < 50


# ------------------------------------------------------------------
# Test: image comparison
# ------------------------------------------------------------------


class TestCompareImages:
    def test_identical_images(
        self, vk: VisionKit, solid_red_image: Image.Image
    ) -> None:
        score = vk.compare_images(solid_red_image, solid_red_image)
        assert score == 1.0

    def test_different_images(self, vk: VisionKit) -> None:
        black = Image.new("RGB", (100, 100), color=(0, 0, 0))
        white = Image.new("RGB", (100, 100), color=(255, 255, 255))
        score = vk.compare_images(black, white)
        assert score == 0.0


# ------------------------------------------------------------------
# Test: filters
# ------------------------------------------------------------------


class TestFilters:
    def test_grayscale_filter(
        self, vk: VisionKit, solid_red_image: Image.Image
    ) -> None:
        gray = vk.apply_filter(solid_red_image, "grayscale")
        arr = np.array(gray)
        # All channels should be equal after grayscale
        assert np.array_equal(arr[:, :, 0], arr[:, :, 1])
        assert np.array_equal(arr[:, :, 1], arr[:, :, 2])

    def test_unknown_filter_raises(
        self, vk: VisionKit, solid_red_image: Image.Image
    ) -> None:
        with pytest.raises(ValueError, match="Unknown filter"):
            vk.apply_filter(solid_red_image, "nonexistent")

    def test_sepia_filter(
        self, vk: VisionKit, solid_red_image: Image.Image
    ) -> None:
        sepia = vk.apply_filter(solid_red_image, "sepia")
        assert sepia.size == solid_red_image.size
        arr = np.array(sepia)
        # Sepia should produce a warm tone (red > green > blue)
        assert arr[:, :, 0].mean() >= arr[:, :, 1].mean()
        assert arr[:, :, 1].mean() >= arr[:, :, 2].mean()
