"""VisionKit configuration via environment variables and pydantic."""

from __future__ import annotations

import logging
import os
from typing import Literal

from pydantic import BaseModel, Field


class VisionKitConfig(BaseModel):
    """Configuration for VisionKit, loaded from environment variables."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity level.",
    )
    max_image_size: int = Field(
        default=4096,
        gt=0,
        description="Maximum allowed image dimension (width or height) in pixels.",
    )

    @classmethod
    def from_env(cls) -> VisionKitConfig:
        """Create a config instance from environment variables."""
        return cls(
            log_level=os.getenv("LOG_LEVEL", "INFO"),  # type: ignore[arg-type]
            max_image_size=int(os.getenv("MAX_IMAGE_SIZE", "4096")),
        )

    def configure_logging(self) -> logging.Logger:
        """Set up and return a logger based on the current config."""
        logger = logging.getLogger("visionkit")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(getattr(logging, self.log_level))
        return logger
