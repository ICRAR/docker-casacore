#!/bin/bash
. /code/venv/bin/activate
export HOME=/scratch
export IPYTHONDIR=$HOME/.ipython
export MPLCONFIGDIR=$HOME
export MPLBACKEND=TKAgg
$1