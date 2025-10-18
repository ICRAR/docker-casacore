#!/bin/bash
. /code/venv/bin/activate
scp -r /code/test /scratch
scp -r /code/demo /scratch
export HOME=/scratch
export IPYTHONDIR=$HOME/.ipython
export MPLCONFIGDIR=$HOME
export MPLBACKEND=TKAgg
$1