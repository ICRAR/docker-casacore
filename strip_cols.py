"""
This module provides a test script for evaluating the performance and accuracy of column-wise compression
using the Adios2StMan storage manager in casacore tables. It generates synthetic complex data, writes it to
a table with configurable compression settings for the real and imaginary parts, and measures the compression
ratio and read/write times. The script also supports plotting the difference between original and compressed
data for visual inspection.

Key Features:
- Configurable compressor and accuracy for each column via command-line arguments.
- Synthetic data generation with user-defined shape.
- Writes original, real, and imaginary parts to separate columns with optional compression.
- Measures and prints compression ratios and timing statistics for write and read operations.
- Optionally plots histograms of differences between original and compressed data.

Dependencies:
- casacore.tables
- numpy
- adios2 (with mgard, sz and zfp)
- matplotlib

Usage:
  python strip_cols.py [--filename DIR] [--datacol=CORRECTED_DATA]
"""
import argparse
import array
import json
import sys

import functools
import operator
import os, time, shutil
from matplotlib import pyplot as plt

from casacore.tables import (
  makearrcoldesc,
  makescacoldesc,
  table,
  maketabdesc,
  makedminfo)
import numpy as np
import adios2
from contextlib import contextmanager
from math import sqrt


def run(DATACOL=DATACOL,FILENAME=FILENAME)-> tuple:
  """Write and read the table and create a copy of the DATA column from the DATA
  """
  tb=table(FILENAME,readonly=False)

  print(f"  MS File: {FILENAME} if COPY_ADIOS is in place will remove COPY_DATA & {DATACOL} columns")
  print()


  if 'COPY_ADIOS' in tb.colnames():
   if 'COPY_DATA' in tb.colnames():
      print('Remove old standard COPY_DATA and the original data column: ',DATACOL)
      tb.removecols('COPY_DATA')
      for DC in DATACOL.split(','):
          if DC in tb.colnames():
              tb.removecols(DC)
          else:
              print(f'No {DC} - not removing that')
   else:
      print('No COPY_DATA - taking no action')
  else:
      print('No COPY_ADIOS - taking no action')
      
  tb.close()


HISTORY=True
if HISTORY==True:
  import time
  tb=tables.table(FILENAME+'/HISTORY',readonly=False)
  n=tb.nrows() #n=-1 # Stick remarks at the end. Could loose important info ..
  tb.addrows(nrows=1)
  #d=t2.getcol('CLI_COMMAND')#,nrow=(t2.nrows()-1))
  #n=d['shape'];n[0]+=1;d['shape']=n
  #for n in range(d['shape'][0]):
  #  if d['array'][n]=='': break
  #d['array'][n]=' '.join(sysargvIn)
  #t2.putcol('CLI_COMMAND',d)
  d=tb.getcol('TIME')
  mjd=time.time()/3600/24 +40588-0.5  # Convert 01/01/1970 to MJD
  d[n]=mjd
  tb.putcol('TIME',d)
  d=tb.getcol('MESSAGE')
  d[n]=f'Removed COPY_DATA and {DATACOL}'
  tb.putcol('MESSAGE',d)
  d=tb.getcol('ORIGIN')
  d[n]=sys.argv[0]#+':'+k[-1]
  tb.putcol('ORIGIN',d)
  tb.close()
  
if __name__ == "__main__":  
  parser = argparse.ArgumentParser(description=
                                   'Test the column-wise compression using the Adios2StMan storage manager in casacore tables')
  parser.add_argument("--filename", type=str, default=FILENAME, help="MS filename")
  parser.add_argument("--datacol", type=str, default=DATACOL, help="Data Column")

  DELETEOLD=True
  args = parser.parse_args()
  if args.filename != FILENAME: 
    FILENAME = args.filename
  if args.datacol != DATACOL: 
    DATACOL = args.datacol

    run(DATACOL=DATACOL,FILENAME=FILENAME)
  sys.exit()
  
