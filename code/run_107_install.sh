#!/bin/sh
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}   # repository root
# CoSpar (Wang et al. 2022) in its own environment, so the analysis environment's pins stay untouched.
cd $ROOT
env/bin/python -m venv ext/env_cospar
ext/env_cospar/bin/python -m pip install -q --upgrade pip
ext/env_cospar/bin/python -m pip install cospar
ext/env_cospar/bin/python -c "import cospar as cs, scanpy, ot; print('cospar', cs.__version__, 'scanpy', scanpy.__version__, 'pot', ot.__version__)"
echo "exit $?"
