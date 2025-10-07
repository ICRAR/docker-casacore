#!/bin/bash
. /code/venv/bin/activate
scp test_addColumn.py /scratch
export HOME=/scratch
export IPYTHONDIR=$HOME/.ipython
export MPLCONFIGDIR=$HOME
export MPLBACKEND=TKAgg
$1