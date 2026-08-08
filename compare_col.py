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
  python create_compressed.py [--compressor COMP] [--accuracy ACC] [--filename DIR]
"""
import argparse
import array
import json
import sys
import time # why is this failing?

import functools
import operator
import os, time, shutil
from matplotlib import pyplot as pl

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

def get_size(msdir):
    size = os.path.getsize(msdir)
    for root, dirs, files in os.walk(msdir):
        size += sum(os.path.getsize(os.path.join(root, name))
                    for name in files + dirs)
    return size

# various settings
FILENAME = "1197634368.ms";
DATA1 = "DATA";
DATA2 = "COPY_ADIOS";
STEPS = 0
DEBUG=True

def compare_ADIOS_column(DATA1=DATA1,DATA2=DATA2,FILENAME=FILENAME,STEPS=STEPS)-> tuple:
  """Compare two column values
  """
  
  import time
  tb=table(FILENAME,readonly=False)
  print(f'Opening data column {DATA1}')
  try:
    SHAP=np.array(tb.getcol(DATA1,nrow=1).shape)
  except:
    print('No Column',DATA1)
    exit()
  try:
    SHAP2=np.array(tb.getcol(DATA2,nrow=1).shape)
  except:
    print('No Column',DATA2)
    exit()
  if np.sum(SHAP)!=np.sum(SHAP2): 
    print(f'Shape of {DATA1} does not match {DATA2}: {SHAP[1:]}:{SHAP2[1:]}')
    exit()
  SHAP[0]=tb.nrows()
  size = functools.reduce(operator.mul, SHAP[1:], SHAP[0] * 8)

  print("Settings:")
  print(f"  For {FILENAME} compare columns: {DATA1} {DATA2}")
  print()

  cell_shape=SHAP[1:]

  if STEPS: # not equal zero
    steps=STEPS
    Nbase=int(SHAP[0]/steps)
    if steps<0:
        steps=len(np.unique(tb.getcol('TIME')))
        Nbase=int(SHAP[0]/steps)
    if (Nbase*steps!=SHAP[0]):
        steps = find_smallest_divisor(SHAP[0])    
        Nbase=SHAP[0]/steps
    print(f'Using {steps} steps to write and read new columns in steps of {Nbase}. {SHAP[0]} should equal {Nbase*steps}')
  else:
    steps=1
    Nbase=SHAP[0]
    print(f'Using one step for writing and reading new columns.')
  tot_tic = time.time()
  mem_budget=8*Nbase*SHAP[1]*SHAP[2]/1e9
  mem_steps=0
  print('Expected Memory footprint few* %.2fGB'%(mem_budget))
  for n in range(0,steps,args.report):
        print('Read %d/%d\t'%(n,steps))
        tic=time.time()
        vis=tb.getcol(DATA1,nrow=Nbase,startrow=n*Nbase)
        tsteps = time.time()-tic
        print(f'Read {Nbase} compressed complex visibilities from reference column in {tsteps:.3f}s')
        data=tb.getcol(DATA2,nrow=Nbase,startrow=n*Nbase)
        a1=tb.getcol('ANTENNA1',nrow=Nbase,startrow=n*Nbase)
        a2=tb.getcol('ANTENNA2',nrow=Nbase,startrow=n*Nbase)
        fg=tb.getcol('FLAG',nrow=Nbase,startrow=n*Nbase)
        tsteps = time.time()-tic
        print(f'Read {Nbase} from other column in {tsteps:.3f}s')
        #I=np.where(flag==False)
        vis *=(1-fg) # zero flagged data
        data*=(1-fg)
        I=np.where((a1!=a2))[0]
        if DEBUG: print('About to calc StdDev')
        readback_orig=np.nanstd(data[I])
        readback=np.nanstd(vis[I])
        data-=vis
        print('Data Difference:',np.nanmax(np.abs(data[I])),np.nanstd(data[I]),'StdDev',readback_orig,readback)
        # End step loop
  t_seq=tb.getdminfo(DATA2)["SEQNR"]
  a_seq=tb.getdminfo(DATA1)["SEQNR"]
  tb.close()
  tsteps = time.time()-tot_tic
  if (steps>1): print(f'Total Read/Write compressed complex visibilities from COPY_ADIOS/COPY_DATA column in {tsteps:.3f}s')

  try: # the ASKAP MS do not generate a separate table
      t_on_disk_size = get_size(f'{FILENAME}/table.f{t_seq}_TSM1')
  except:
      print('Guessing default disk size for COPY_DATA as not separate directory')
      t_on_disk_size = SHAP[0]*SHAP[1]*SHAP[2]*8
  a_on_disk_size = get_size(f'{FILENAME}/table.f{a_seq}.bp')
  rat_on_disk_size=100.*a_on_disk_size/t_on_disk_size
  print(f'Native size: {t_on_disk_size} Compressed size: {a_on_disk_size} or {rat_on_disk_size:.1f}%')
     
if __name__ == "__main__":  
  parser = argparse.ArgumentParser(description='Test the column-wise compression using the Adios2StMan storage manager in casacore tables')
  parser.add_argument("--memlim", type=int, default=0, help="An attempt to manage memory limits")
  parser.add_argument("--filename", type=str, default=FILENAME, help="MS filename")
  parser.add_argument("--data1", type=str, default=DATA1, help="Data Column 1")
  parser.add_argument("--data2", type=str, default=DATA2, help="Data Column 2")
  parser.add_argument("--steps", type=int, default=0, help="Write a STEPS column in this number of steps.")
  parser.add_argument("--report", type=int, default=10, help="Reporting interval")

  args = parser.parse_args()
  if args.steps:
    STEPS = args.steps #(STEPS==False) # Swap the setting
  if args.filename != FILENAME: 
    FILENAME = args.filename
  if args.data1 != DATA1: 
    DATA1 = args.data1
  if args.data1 != DATA2: 
    DATA2 = args.data2
  compare_ADIOS_column(DATA1=DATA1,DATA2=DATA2,FILENAME=FILENAME,STEPS=STEPS)
  sys.exit()
  
