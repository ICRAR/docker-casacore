#!/bin/bash
. /code/venv/bin/activate
scp /code/test/* /scratch
export HOME=/scratch
export IPYTHONDIR=$HOME/.ipython
export MPLCONFIGDIR=$HOME
export MPLBACKEND=TKAgg
$1