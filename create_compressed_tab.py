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
    size = os.path.getsize(msdir)
    for root, dirs, files in os.walk(msdir):
        size += sum(os.path.getsize(os.path.join(root, name))
                    for name in files + dirs)
    return size

def mk_plot(vis,n):
    try:
        pl.clf()
        pl.hist(np.abs(vis),bins=30)
        pl.ylabel('No.')
        pl.xlabel(r'Abs $\Delta$ Error')
        pl.savefig('tmp.%3d.png'%(n))
    except:
        print("Couldn't write plot")


# various settings
COMPRESSORS = ["mgard", "mgard_complex", "zfp", "sz", "None"];
COMPRESSOR = "mgard";
MODE = 'ABS';
ACCURACY = "0.1";
FILENAME = "1197634368.ms";
DATACOL = "CORRECTED_DATA";
DELETEOLD = True
SKIPDATA = False
PLOT = False
STEPS = 0

def run(DATACOL=DATACOL,FILENAME=FILENAME,
        STEPS=STEPS,ACCURACY=ACCURACY,COMPRESSOR=COMPRESSOR,
        DELETEOLD=DELETEOLD,PLOT=PLOT,SKIPDATA=SKIPDATA)-> tuple:
  """Write and read the table and create a copy of the DATA column from the DATA
  """
  import time # why is this failing?
  tb=table(FILENAME)
  try:
      print(f'Opening data column {DATACOL}')
      n=tb.getcol(DATACOL,nrow=1)
      vtype=str(n.dtype)[:-2]
      SHAP=np.array(n.shape)
  except:
      #print('No datacolumn %s trying "DATA"'%(DATACOL))
      DATACOL = 'DATA'
      print(f'Opening data column {DATACOL}')
      n=tb.getcol(DATACOL,nrow=1)
      vtype=str(n.dtype)[:-2]
      SHAP=np.array(n.shape)
  SHAP[0]=tb.nrows()
  size = functools.reduce(operator.mul, SHAP[1:], SHAP[0] * 8)

  print("Settings:")
  print(f"  Compressor for {DATACOL} column: {COMPRESSOR} (Accuracy: {ACCURACY})")
  print(f"  Data shape: {SHAP}")
  print(f"  MS File: {FILENAME}")
  print()

  vtype='complex'
  cell_shape=SHAP[1:]
  Atabdesc = maketabdesc(
        (makearrcoldesc('COPY', '',
            valuetype=vtype, shape=cell_shape,
                        datamanagergroup='group0', #%4d'%(np.random.randint(0,9999)),
                        datamanagertype='Adios2StMan' ),
         ))
  if COMPRESSOR=="None":
    Adminfo = makedminfo(
        Atabdesc,
        {
            'group0': {
                'OPERATORPARAMS': {
                    'COPY': {
                        'lossless' : 'Huffman_Zstd'}
               } } } )      
  else:
    Adminfo = makedminfo(
        Atabdesc,
        {
            'group0': {
                'OPERATORPARAMS': {
                    'COPY': {
                        'Operator': COMPRESSOR,
                        'mode': MODE,
                        'Accuracy': str(ACCURACY),
                    #    'lossless' : 'Huffman_Zstd'
                    }
               } } } )

  Ttabdesc = maketabdesc(
        (makearrcoldesc('COPY', '',
            valuetype=vtype, shape=cell_shape,
                        datamanagergroup='group1', #%4d'%(np.random.randint(0,9999)),
                        datamanagertype='TiledShapeStMan' ),
         ))
  Tdminfo = makedminfo(
        Ttabdesc,
        {
            'group1': {
                'OPERATORPARAMS': {
                    'COPY': {}
               } } } )

  outname=f'{FILENAME}.{COMPRESSOR}.{ACCURACY}.adios'
  outname=outname.replace('.ms/','').replace('.ms','')
  if (os.path.isdir(outname)==True): shutil.rmtree(outname)
  
  tb2=table(outname,Atabdesc,dminfo=Adminfo)
  tb2.addrows(SHAP[0])
      
  if SKIPDATA==False:
      outname2=outname.replace('adios','tab')
      if (os.path.isdir(outname2)==True): shutil.rmtree(outname2)
      tb3=table(outname2,Ttabdesc,dminfo=Tdminfo)
      tb3.addrows(SHAP[0])

  if STEPS: # not equal zero
    steps=STEPS
    Nbase=int(SHAP[0]/steps)
    if steps<0:
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
  tot_tic = time.time()
  for n in range(steps):
    print('Write %d/%d\t'%(n,steps))
    tic=time.time()
    vis=tb.getcol(DATACOL,nrow=Nbase,startrow=n*Nbase)
    tsteps = time.time()-tic
    print(f'Read {Nbase} compressed {vtype} visibilities from {DATACOL} column in {tsteps:.3f}s')
    s=vis.shape
    if (COMPRESSOR == "mgard_complex")|(COMPRESSOR == "mgard"):
        vis=vis.reshape(-1)
        Inan=np.where(np.isnan(vis)==True)[0]
        vis[Inan]=0
        vis=vis.reshape(s)
    tic = time.time()
    tb2.putcol('COPY',vis,nrow=Nbase,startrow=n*Nbase)
    tsteps = time.time()-tic
    print(f'Wrote {Nbase} compressed {vtype} visibilities to ADIOS column in {tsteps:.3f}s')
  tsteps = time.time()-tot_tic
  if (steps>1): print(f'Total Read/Write compressed {vtype} visibilities from {DATACOL}/COPY_ADIOS column in {tsteps:.3f}s')
  tb2.close()

  if SKIPDATA==False:
      tb2=table(outname)
      for n in range(steps):
         vis=tb2.getcol('COPY',nrow=Nbase,startrow=n*Nbase)
         tb3.putcol('COPY',vis,nrow=Nbase,startrow=n*Nbase)
         if n==0:
             data=vis-tb.getcol(DATACOL,nrow=Nbase,startrow=n*Nbase)
             a1=tb.getcol('ANTENNA1',nrow=Nbase,startrow=n*Nbase)
             a2=tb.getcol('ANTENNA2',nrow=Nbase,startrow=n*Nbase)
             I=np.where(a1!=a2)[0]
             print('Data Differences: ',
                   np.nanstd(vis[I]),np.nanmax(np.abs(data[I])),np.nanstd(data[I]))
      tb3.close()
      tb2.close()
      t_on_disk_size = get_size(outname2)
      a_on_disk_size = get_size(outname)
      rat_on_disk_size=100.*a_on_disk_size/t_on_disk_size
      print(f'Native size: {t_on_disk_size} Compressed size: {a_on_disk_size} or {rat_on_disk_size:.1f}%')
  else:
      print('Skipping back-conversion to COPY_DATA')
  tb.close()
  #print(f'ORIG write time: {tnocomp_complex:.3f}')
  #print(f'REAL[{COMPRESSOR1}] compression and write time: {tcomp_real:.3f}')
  #print(f'IMAG[{COMPRESSOR2}] compression and write time: {tcomp_imag:.3f}\n')
  #print('Total compression and write time: '
  #      f'{(tcomp_real+tcomp_imag):.3f} ({((tcomp_real+tcomp_imag)/tnocomp_complex):.1f}x)\n\n')
  #
  #print(f'ORIG read time: {(tread_complex):.3f} s')
  #print(f'REAL[{COMPRESSOR1}] decompression and read time: {(tdecomp_real):.3f} s')
  #print(f'REAL compression ratio: {size / r_on_disk_size:.2f}')
  #print(f'IMAG[{COMPRESSOR2}] decompression and read time: {(tdecomp_imag):.3f} s')
  #print(f'IMAG compression ratio: {size / i_on_disk_size:.2f}\n')
  #print('Total decompression and read time: '
  #      f'{(tdecomp_real+tdecomp_imag):.3f} s ({((tdecomp_real+tdecomp_imag)/tread_complex):.1f}x)\n\n')
  
  HISTORY=False
  if HISTORY==True:
    import time
    tb=table(f"{FILENAME}/HISTORY",readonly=False)
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
    d[n]=f'Applied compressor {COMPRESSOR} with accuracy {ACCURACY} to make COPY_ADIOS (and then COPY_DATA) from {DATACOL}'
    print('Writing HISTORY: ',d[n])
    tb.putcol('MESSAGE',d)
    d=tb.getcol('ORIGIN')
    d[n]=sys.argv[0]#+':'+k[-1]
    tb.putcol('ORIGIN',d)
    d=tb.getcol('ORIGIN')
    d[n]=sys.argv[0]#+':'+k[-1]
    tb.putcol('ORIGIN',d)
    tb.close()
  
if __name__ == "__main__":  
  parser = argparse.ArgumentParser(description=
                                   'Test the column-wise compression using the Adios2StMan storage manager in casacore tables')
  parser.add_argument("--compressor", type=str, default=COMPRESSOR, help="Global data compressor")
  parser.add_argument("--accuracy", type=str, default=ACCURACY, help="Global accuracy for data columns")
  parser.add_argument("--filename", type=str, default=FILENAME, help="MS filename")
  parser.add_argument("--datacol", type=str, default=DATACOL, help="Data Column")
  parser.add_argument("--steps", type=int, default=0, help="Write a STEPS column in this number of steps.")
  parser.add_argument("--plot", action='store_true', help="(False) Plot comparison histograms.")
  parser.add_argument("--skipdata", action='store_true', help="(False) Dont back convert ADIOS to TileStMan.")
  parser.add_argument("--reuse", action='store_true', help="(False) delete and remake columns")

  DELETEOLD=True
  args = parser.parse_args()
  if args.accuracy != ACCURACY:
      ACCURACY = args.accuracy
  if args.steps:
    STEPS = args.steps #(STEPS==False) # Swap the setting
  if args.reuse:
    print('Swapping delete flag ',DELETEOLD)
    DELETEOLD = (args.reuse==False) # Swap the setting for delete
  if args.skipdata:
    SKIPDATA = True
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
  #print(FILENAME,DATACOL,DELETEOLD,COMPRESSOR,ACCURACY)    
  run(DATACOL=DATACOL,FILENAME=FILENAME,
      STEPS=STEPS,ACCURACY=ACCURACY,COMPRESSOR=COMPRESSOR,
      DELETEOLD=DELETEOLD,PLOT=PLOT,SKIPDATA=SKIPDATA)
  sys.exit()
  
