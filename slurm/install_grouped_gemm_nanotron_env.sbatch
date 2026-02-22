#!/bin/bash
#SBATCH --job-name=inst_ggemm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:1
#SBATCH --account=EUHPC_A04_051
#SBATCH --output=logs/ggemm_%j.out
#SBATCH --error=logs/ggemm_%j.err

module purge
module load python/3.11
module load cuda/12.1
module load gcc/12

# (προαιρετικό) ίδιο libstdc++ fix
GCC_HOME=$(dirname $(dirname $(which g++)))
export LD_PRELOAD="$GCC_HOME/lib64/libstdc++.so.6"
export LD_LIBRARY_PATH="$GCC_HOME/lib64:$LD_LIBRARY_PATH"

source /leonardo_scratch/large/userexternal/mpeppas0/venvs/nanotron_env/bin/activate
mkdir -p logs

python -m pip install -U pip setuptools wheel ninja

# Για να μην κάνει OOM από parallel compile
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1

python -m pip install --no-build-isolation \
  "git+https://github.com/fanshiqing/grouped_gemm@main"

python -c "import grouped_gemm; print('grouped_gemm OK')"
