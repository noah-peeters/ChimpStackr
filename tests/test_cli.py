"""
Test the CLI interface.
"""
import os, sys, subprocess, tempfile

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytest
import numpy as np
import cv2


class TestCLI:
    def test_cli_stack_only(self):
        """CLI stack-only mode should produce an output file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "src.cli",
                    "--input", "tests/low_res_images/*.jpg",
                    "--output", output_path,
                    "--pyramid-levels", "4",
                ],
                capture_output=True,
                text=True,
                cwd=parentdir,
                timeout=120,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.isfile(output_path)
            assert os.path.getsize(output_path) > 0

            img = cv2.imread(output_path)
            assert img is not None
            assert img.shape == (500, 750, 3)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_cli_align_and_stack(self):
        """CLI align+stack mode should produce an output file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "src.cli",
                    "--input", "tests/low_res_images/*.jpg",
                    "--output", output_path,
                    "--align",
                    "--pyramid-levels", "4",
                ],
                capture_output=True,
                text=True,
                cwd=parentdir,
                timeout=120,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.isfile(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_cli_quality_report(self):
        """CLI with --quality-report should print sharpness info."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "src.cli",
                    "--input", "tests/low_res_images/*.jpg",
                    "--output", output_path,
                    "--pyramid-levels", "4",
                    "--quality-report",
                ],
                capture_output=True,
                text=True,
                cwd=parentdir,
                timeout=120,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert "sharpness" in result.stdout.lower()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_cli_insufficient_images(self):
        """CLI should fail with fewer than 2 images."""
        result = subprocess.run(
            [
                sys.executable, "-m", "src.cli",
                "--input", "tests/low_res_images/DSC_0356.jpg",
                "--output", "/tmp/test_single.jpg",
            ],
            capture_output=True,
            text=True,
            cwd=parentdir,
            timeout=30,
        )
        assert result.returncode != 0

    def test_cli_custom_params(self):
        """CLI with custom kernel size and alignment ref should work."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "src.cli",
                    "--input", "tests/low_res_images/*.jpg",
                    "--output", output_path,
                    "--align",
                    "--kernel-size", "4",
                    "--pyramid-levels", "4",
                    "--alignment-ref", "middle",
                ],
                capture_output=True,
                text=True,
                cwd=parentdir,
                timeout=120,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
