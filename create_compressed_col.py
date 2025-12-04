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

import functools
import operator
import os, time
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

def find_smallest_divisor(num:int) -> int:
    """Find the smalles divisor of num greater or equal to 3.

    Args:
        num (int): The number to be considered

    Returns:
        int: The smallest number >=3 that divides num withoput rest.
  """
    for i in range(3, int(sqrt(num)) + 1):
        if num % i == 0:
            return i
    return num

def get_size(msdir):
    size = 0
    for root, dirs, files in os.walk(msdir):
        size += sum(os.path.getsize(os.path.join(root, name))
                    for name in files + dirs)
    return size


# various settings
COMPRESSORS = ["mgard", "mgard_complex", "zfp", "sz", "None"];
COMPRESSOR = "mgard";
MODE = 'ABS';
ACCURACY = "0.1";
FILENAME = "1197634368.ms";
DATACOL = "CORRECTED_DATA";
PLOT = False
STEPS = False

def run()-> tuple:
  """Write and read the table and create a copy of the DATA column from the DATA
  """
  tb=table(FILENAME,readonly=False)
  try:
      SHAP=np.array(tb.getcol(DATACOL,nrow=1).shape)
  except:
      #print('No datacolumn %s trying "DATA"'%(DATACOL))
      DATACOL = 'DATA'
      SHAP=np.array(tb.getcol(DATACOL,nrow=1).shape)
  SHAP[0]=tb.nrows()
  size = functools.reduce(operator.mul, SHAP[1:], SHAP[0] * 8)

  print("Settings:")
  print(f"  Compressor for {DATACOL} column: {COMPRESSOR} (Accuracy: {ACCURACY})")
  print(f"  Data shape: {SHAP}")
  print(f"  Output directory: {FILENAME}")
  print()

  cell_shape=SHAP[1:]
  Atabdesc = maketabdesc(
        (makearrcoldesc('COPY_ADIOS', '',
            valuetype='complex', shape=cell_shape,
            datamanagergroup='group0', datamanagertype='Adios2StMan' ),
         ))
  Adminfo = makedminfo(
        Atabdesc,
        {
            'group0': {
                'OPERATORPARAMS': {
                    'COPY_ADIOS': {
                        'Operator': COMPRESSOR,
                        'mode': MODE,
                        'Accuracy': str(ACCURACY)}
               } } } )

  Ttabdesc = maketabdesc(
        (makearrcoldesc('COPY_DATA', '',
            valuetype='complex', shape=cell_shape,
            datamanagergroup='group1', datamanagertype='TiledShapeStMan' ),
         ))
  Tdminfo = makedminfo(
        Ttabdesc,
        {
            'group1': {
                'OPERATORPARAMS': {
                    'COPY_DATA': {
                        'Operator': COMPRESSOR,
                        'mode': MODE,
                        'Accuracy': str(ACCURACY)}
               } } } )

  if 'COPY_ADIOS' in tb.colnames():
      print('Remove old adios COPY')
      tb.removecols('COPY_ADIOS')
  if 'COPY_DATA' in tb.colnames():
      print('Remove old standard COPY_DATA')
      tb.removecols('COPY_DATA')
  tb.addcols(Atabdesc,dminfo=Adminfo)
  tb.addcols(Ttabdesc,dminfo=Tdminfo)

  if STEPS:
    steps=len(np.unique(tb.getcol('TIME')))
    Nbase=int(SHAP[0]/steps)
    if (Nbase*steps!=SHAP[0]):
        steps = find_smallest_divisor(SHAP[0])    
        Nbase=SHAP[0]/steps
    print(f'Using {steps} steps to write and read new columns in steps of {Nbase}.')
  else:
    steps=1
    Nbase=SHAP[0]
    print(f'Using one step for writing and reading new columns.')
  for n in range(steps):
    print('Write %d/%d\t'%(n,steps))
    tic = time.time()
    vis=tb.getcol(DATACOL,nrow=Nbase,startrow=n*Nbase)
    tsteps = time.time()-tic
    print(f'Read {Nbase} compressed complex visibilities from {DATACOL} column in {tsteps:.3f}s')
    s=vis.shape
    if COMPRESSOR == "mgard_complex":
        vis=vis.reshape(-1)
        Inan=np.where(np.isnan(vis)==True)[0]
        vis[Inan]=0
        vis=vis.reshape(s)
    tic = time.time()
    tb.putcol('COPY_ADIOS',vis,nrow=Nbase,startrow=n*Nbase)
    tsteps = time.time()-tic
    print(f'Wrote {Nbase} compressed complex visibilities to ADIOS column in {tsteps:.3f}s')
  tb.close()

  tb=table(FILENAME,readonly=False)
  for n in range(steps):
    print('Read %d/%d\t'%(n,steps))
    #tic=time.time()
    #data=t.getcol('COPY_DATA',nrow=Nbase,startrow=n*Nbase)
    #tsteps = time.time()-tic
    #print(f'Read {Nbase} compressed complex visibilities from COPY_DATA column in {tsteps:.3f}s')
    tic=time.time()
    vis=tb.getcol('COPY_ADIOS',nrow=Nbase,startrow=n*Nbase)
    tsteps = time.time()-tic
    print(f'Read {Nbase} compressed complex visibilities from COPY_ADIOS column in {tsteps:.3f}s')
    tic = time.time()
    tb.putcol('COPY_DATA',vis,nrow=Nbase,startrow=n*Nbase)
    tsteps = time.time()-tic
    print(f'Wrote {Nbase} compressed complex visibilities to TILED column in {tsteps:.3f}s')
  a_seq=tb.getdminfo("COPY_ADIOS")["SEQNR"]
  t_seq=tb.getdminfo("COPY_DATA")["SEQNR"]
  tb.close()

  t_on_disk_size = get_size(f'{FILENAME}/table.f{t_seq}i')
  #print(f'ORIG write time: {tnocomp_complex:.3f}')
  #print(f'REAL[{COMPRESSOR1}] compression and write time: {tcomp_real:.3f}')
  a_on_disk_size = get_size(f'{FILENAME}/table.f{a_seq}.bp')
  #print(f'IMAG[{COMPRESSOR2}] compression and write time: {tcomp_imag:.3f}\n')
  #print('Total compression and write time: '
  #      f'{(tcomp_real+tcomp_imag):.3f} ({((tcomp_real+tcomp_imag)/tnocomp_complex):.1f}x)\n\n')

  #print(f'ORIG read time: {(tread_complex):.3f} s')
  #print(f'REAL[{COMPRESSOR1}] decompression and read time: {(tdecomp_real):.3f} s')
  #print(f'REAL compression ratio: {size / r_on_disk_size:.2f}')
  #print(f'IMAG[{COMPRESSOR2}] decompression and read time: {(tdecomp_imag):.3f} s')
  #print(f'IMAG compression ratio: {size / i_on_disk_size:.2f}\n')
  #print('Total decompression and read time: '
  #      f'{(tdecomp_real+tdecomp_imag):.3f} s ({((tdecomp_real+tdecomp_imag)/tread_complex):.1f}x)\n\n')
  #if PLOT:
  #  plot(vis, visr, visi, cvis)
    
if __name__ == "__main__":  
  parser = argparse.ArgumentParser(description=
                                   'Test the column-wise compression using the Adios2StMan storage manager in casacore tables')
  parser.add_argument("--compressor", type=str, default=COMPRESSOR, help="Global data compressor")
  parser.add_argument("--accuracy", type=str, default=ACCURACY, help="Global accuracy for data columns")
  parser.add_argument("--filename", type=str, default=FILENAME, help="Output filename")
  parser.add_argument("--datacol", type=str, default=DATACOL, help="Data Column")
  parser.add_argument("--steps", action='store_true', help="(False) Write a STEPS column in steps.")
  parser.add_argument("--plot", action='store_true', help="(False) Plot comparison histograms.")

  args = parser.parse_args()
  if args.accuracy != ACCURACY:
      ACCURACY = args.accuracy
  if args.steps:
    STEPS = (STEPS==False) # Swap the setting
  if args.plot:
    PLOT = True
  if args.filename != FILENAME: 
    FILENAME = args.filename
  if args.datacol != DATACOL: 
    DATACOL = args.datacol
  if args.compressor not in COMPRESSORS:
      print(f"compressor argument needs to be one of {COMPRESSORS}")
      sys.exit()
  else:
      COMPRESSOR = args.compressor
  print(FILENAME,DATACOL)    
  run()
  sys.exit()
  
