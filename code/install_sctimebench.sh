#!/bin/sh
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}   # repository root
# One environment for the scTimeBench-wrapped methods (scNODE, MIOFlow, PISDE, scIMF, Squidiff).
# The methods' own pinned environments use torch 1.13-2.4, which has no kernels for the RTX 5090;
# this uses scNODE's pinned stack (torch 2.8, CUDA 12.8) for all and adds the other methods' packages.
set -e
cd $ROOT/ext
export PIP_CACHE_DIR=$ROOT/ext/pipcache TMPDIR=$ROOT/ext/tmp
mkdir -p $TMPDIR
rm -rf env_stb
# a plain virtual environment on the Python 3.11 interpreter of env_ot (conda would need its
# channel terms accepted); none of env_ot's packages are shared
$ROOT/env_ot/bin/python -m venv $ROOT/ext/env_stb
P=$ROOT/ext/env_stb/bin/pip
$P install --upgrade pip
$P install -r scTimeBench/methods/scNODE/scNODE_module/requirements.txt
$P install POT torchsde trampoline boltons MIOFlow Squidiff
$P install -e scTimeBench
$P freeze > env_stb_freeze.txt
$ROOT/ext/env_stb/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
