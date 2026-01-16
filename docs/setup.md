# Setup with uv

### Prerequisites
Install [uv](https://docs.astral.sh/uv/) if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation
```bash
git clone --recurse-submodules https://github.com/arthurfenderbucker/open_perception.git
cd open_perception
uv sync
# Install third party models
git submodule update --init --recursive
uv pip install -e ./third_party/segment-anything-2-real-time
```

Download the [pre-trained models](../checkpoints/README.md)

### Running the Pipeline
```bash
uv run python src/main.py
```

### Optional: Installing Additional Dependencies

For specific features, you can install optional dependency groups:

```bash
# For RealSense camera support
uv sync --extra realsense

# For YOLO-World model support
uv sync --extra yoloworld

# For sentence transformers (lookup functionality)
uv sync --extra lookup

# For development tools (pytest, etc.)
uv sync --extra dev

# Install all optional dependencies
uv sync --extra all
```

### Notes on CUDA Support

This project uses PyTorch with CUDA 12.4 support. The uv configuration in `pyproject.toml` already specifies the correct PyTorch index URL. The CUDA libraries will be installed automatically through PyTorch's wheel distribution.

If you encounter issues with CUDA or need specific CUDA toolkit versions for building extensions (like GroundingDINO), you may need to install system-level CUDA packages. Refer to your system's CUDA installation documentation or the [possible_errors.md](./possible_errors.md) for troubleshooting.

### Advanced: Custom Python Version

By default, uv will use the Python version specified in `.python-version` file (3.11.7). If you need to use a different Python version:

```bash
uv python install 3.11.7  # Install specific Python version
uv sync  # Sync dependencies with the specified Python version
```
