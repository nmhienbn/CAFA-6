#!/bin/bash

conda create --solver=libmamba -p $1/rapids-env \
    -c rapidsai -c conda-forge -c nvidia \
    python=3.10 cuda-version=11.8 cudatoolkit=11.8 aria2 -y

conda activate $1/rapids-env

pip uninstall cupy numba -y 2>/dev/null || true

pip install tqdm cupy-cuda11x numba==0.56.4 py-boost==0.4.3 pandas yaml
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11 cuml-cu11 cugraph-cu11