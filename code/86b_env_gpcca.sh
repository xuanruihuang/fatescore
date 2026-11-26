ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}   # repository root
# Builds env_gpcca (PETSc/SLEPc for CellRank's Krylov Schur decomposition), used by 86_terminal_states.py --only-gpcca.
set -e
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY
cd $ROOT
CF=conda-forge
CONDA_REMOTE_MAX_RETRIES=10 CONDA_REMOTE_READ_TIMEOUT_SECS=120 conda create -y -p ./env_gpcca -c $CF --override-channels python=3.11 petsc4py slepc4py 'numpy<2.5'
./env_gpcca/bin/python -c 'import petsc4py, slepc4py, numpy; print(petsc4py.__version__, slepc4py.__version__, numpy.__version__)'
./env_gpcca/bin/pip install cellrank==2.0.7 scvelo==0.3.4 scanpy==1.11.5 anndata==0.12.19 scikit-learn==1.9.1 pandas==2.3.3
./env_gpcca/bin/python -c 'import cellrank, scvelo, petsc4py, slepc4py; print("ok", cellrank.__version__)'
