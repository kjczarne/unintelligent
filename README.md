# Unintelligent

## Installation

First install the package itself, with development dependencies:

```bash
poetry install --with dev
```

Then use `poe` to install the right version of PyTorch and Huggingface libs:

```bash
poe install-pytorch --flavor cuda     # if you have an NVIDIA GPU
poe install-pytorch --flavor no-cuda  # if you have no GPU
poe install-huggingface
```

> 👀 If you need an older version of PyTorch, install the `poetry` package first and then install PyTorch and Huggingface directly using `pip` or `conda`. The `poe` commands I've provided will by default pull the most recent version of the libraries. At the moment Poetry has poor support for alternative builds of packages (with CUDA, without CUDA, who would have time and patience for this mess), so this two-step installation is the only saving grace at the moment.
