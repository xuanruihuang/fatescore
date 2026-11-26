#!/bin/sh
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}   # repository root
# scTimeBench's MIOFlow runner needs the package layout before the 2026 restructure (MIOFlow.utils,
# .models, .train, .geo, .eval) and a training_regimen that takes ot_lambda_global: the last such
# commit is KrishnaswamyLab/MIOFlow 1c6ceb4 (2025-08-12). The PyPI releases do not match
# (0.1.14 is the restructured package; 0.1.1 lacks ot_lambda_global / condition_dim).
export TMPDIR=$ROOT/ext/tmp
P=$ROOT/ext/env_stb/bin/python
$P -m pip install -q phate graphtools torchdiffeq seaborn 2>&1 | tail -3
$P -m pip install -q --no-deps --force-reinstall \
  "https://github.com/KrishnaswamyLab/MIOFlow/archive/1c6ceb49841b17d178d4795439182b57250b467d.tar.gz" 2>&1 | tail -3
$P -c "import MIOFlow.utils, MIOFlow.models, MIOFlow.train, MIOFlow.geo, MIOFlow.eval, inspect; print('MIOFlow import ok', 'ot_lambda_global' in inspect.signature(MIOFlow.train.training_regimen).parameters)"
echo "exit $?"
