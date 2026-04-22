"""
This module provides a test script for evaluating the performance and accuracy of column-wise compression
using the Adios2StMan storage manager in casacore tables. It generates synthetic complex data, writes it to
a table with configurable compression settings for the real and imaginary parts, and measures the compression
ratio and read/write times. The script also supports plotting the difference between original and compressed
data for visual inspection.

Key Features:
- Configurable compressor and accuracy for each column via command-line arguments.
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

def find_smallest_divisor(num:int,mindiv=3) -> int:
    """Find the smalles divisor of num greater or equal to 3.

    Args:
        num (int): The number to be considered

    Returns:
        int: The smallest number >=3 that divides num withoput rest.
  """
    for i in range(mindiv, int(sqrt(num)) + 1):
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
        nm='tmp.%03d.png'%(n)
        pl.hist(np.abs(vis),bins=30)
        pl.ylabel('No.')
        pl.xlabel(r'Abs $\Delta$ Error')
        pl.savefig(nm)
    except:
        print('Cant write plot')

# various settings
COMPRESSORS = ["mgard", "mgard_complex", "zfp", "sz", "None"];
COMPRESSOR = "sz";
MODE = 'ABS';
ACCURACY = "0.4";
FILENAMES = [#'Raw/scienceData.G23_T0_A_01.SB32511.G23_T0_A_01.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_01.SB39697.G23_T0_A_01.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_02.SB39722.G23_T0_A_02.beam17_SL.ms',
 #'Raw/scienceData.G23_T0_A_02.SB40329.G23_T0_A_02.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_03.SB40196.G23_T0_A_03.beam17_SL.ms',
 #'Raw/scienceData.G23_T0_A_03.SB40479.G23_T0_A_03.beam17_SL.ms',
 #'Raw/scienceData.G23_T0_A_04.SB40607.G23_T0_A_04.beam17_SL.ms',
 #'Raw/scienceData.G23_T0_A_05.SB40712.G23_T0_A_05.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H01.SB65177.G23_T0_A_H01.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H02.SB65210.G23_T0_A_H02.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H03.SB65268.G23_T0_A_H03.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H04.SB65297.G23_T0_A_H04.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H05.SB65347.G23_T0_A_H05.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H06.SB65405.G23_T0_A_H06.beam17_SL.ms',
 #'Raw/scienceData.G23_T0_A_H07.SB65453.G23_T0_A_H07.beam17_SL.ms',
 #'Raw/scienceData.G23_T0_A_H08.SB65681.G23_T0_A_H08.beam17_SL.ms',
 #'Raw/scienceData.G23_T0_A_H09.SB65765.G23_T0_A_H09.beam17_SL.ms',
 #'Raw/scienceData.G23_T0_A_H10.SB66032.G23_T0_A_H10.beam17_SL.ms',
 #'Raw/scienceData.G23_T0_A_H11.SB66053.G23_T0_A_H11.beam17_SL.ms',
 #'Raw/scienceData.G23_T0_A_H12.SB66106.G23_T0_A_H12.beam17_SL.ms',
 #'Raw/scienceData.G23_T0_A_H13.SB66134.G23_T0_A_H13.beam17_SL.ms',
 #'Raw/scienceData.G23_T0_B_H06.SB65853.G23_T0_B_H06.beam17_SL.ms'
             ]
FILENAME=FILENAMES[0]
DATACOL = "DATA"
DELETEOLD = False
EXTERNAL = True
PLOT = False
STEPS = -1

def run(DATACOL=DATACOL,FILENAME=FILENAME,
        STEPS=STEPS,ACCURACY=ACCURACY,COMPRESSOR=COMPRESSOR,
        DELETEOLD=DELETEOLD,EXTERNAL=EXTERNAL,PLOT=PLOT)-> tuple:
  """Write and read the table and create a copy of the DATA column from the DATA
  """
  import time # why is this failing?
  tb=table(FILENAME,readonly=False)
  try:
      print(f'Opening data column {DATACOL}')
      SHAP=np.array(tb.getcol(DATACOL,nrow=1).shape)
  except:
      #print('No datacolumn %s trying "DATA"'%(DATACOL))
      DATACOL = 'DATA'
      print(f'Opening data column {DATACOL}')
      SHAP=np.array(tb.getcol(DATACOL,nrow=1).shape)
  SHAP[0]=tb.nrows()
  size = functools.reduce(operator.mul, SHAP[1:], SHAP[0] * 8)

  print("Settings:")
  print(f"  Compressor for {DATACOL} column: {COMPRESSOR} (Accuracy: {ACCURACY})")
  print(f"  Data shape: {SHAP}")
  print(f"  MS File: {FILENAME}")
  print()

  cell_shape=SHAP[1:]
  Atabdesc = maketabdesc(
        (makearrcoldesc('COPY_ADIOS', '',
            valuetype='complex', shape=cell_shape,
            comment=f'Copy of {DATACOL} after using {COMPRESSOR} with error bound of {MODE}/{ACCURACY}',
            datamanagergroup='groupA', datamanagertype='Adios2StMan' ),
         ))
  if COMPRESSOR=="None":
    Adminfo = makedminfo(
        Atabdesc,
        {
            'groupA': {
                'OPERATORPARAMS': {
                    'COPY_ADIOS': {
                        'lossless' : 'Huffman_Zstd'}
               } } } )      
  else:
    Adminfo = makedminfo(
        Atabdesc,
        {
            'groupA': {
                'OPERATORPARAMS': {
                    'COPY_ADIOS': {
                        'Operator': COMPRESSOR,
                        'mode': MODE,
                        'Accuracy': str(ACCURACY)}
               } } } )

  Ttabdesc = maketabdesc(
        (makearrcoldesc('COPY_DATA', '',
            valuetype='complex', shape=cell_shape,
            comment=f'Copy of {DATACOL} after using {COMPRESSOR} with error bound of {MODE}/{ACCURACY}',
            datamanagergroup='groupB', datamanagertype='TiledShapeStMan' ),
         ))
  Tdminfo = makedminfo(
        Ttabdesc,
        {
            'groupB': {
                'OPERATORPARAMS': {
                    'COPY_DATA': {}
               } } } )

  if EXTERNAL==False:
   if 'COPY_ADIOS' in tb.colnames():
    if DELETEOLD:
      print('Remove old adios COPY')
      a_seq=tb.getdminfo("COPY_ADIOS")["SEQNR"]
      tb.removecols('COPY_ADIOS')
      print(f'Removing {FILENAME}/table.f{a_seq}.bp')
      shutil.rmtree(f'{FILENAME}/table.f{a_seq}.bp')
      tb.addcols(Atabdesc,dminfo=Adminfo)
    else:
      print('Reusing old adios COPY')
   else:
      tb.addcols(Atabdesc,dminfo=Adminfo)
  else:
      TabName=f'{FILENAME}.{COMPRESSOR}.{ACCURACY}.adios'.replace('.ms/','').replace('.ms','')
      if os.path.isdir(TabName==True): shutil.rmtree(TabName)
      tb2=table(TabName,Atabdesc,dminfo=Adminfo)
      tb2.addrows(SHAP[0])
   
  if 'COPY_DATA' in tb.colnames():
    if DELETEOLD:
      print('Remove old standard COPY_DATA')
      #t_seq=tb.getdminfo("COPY_DATA")["SEQNR"]
      tb.removecols('COPY_DATA')
      #print(f'Removing {FILENAME}/table.f{t_seq}_TSM1')
      #shutil.rmtree(f'{FILENAME}/table.f{t_seq}_TSM1')
      tb.addcols(Ttabdesc,dminfo=Tdminfo)
    else:
      print('Reusing old adios COPY')
  else:
      tb.addcols(Ttabdesc,dminfo=Tdminfo)

  if STEPS: # not equal zero
    steps=STEPS
    Nbase=int(SHAP[0]/steps)
    if steps<0:
        steps=len(np.unique(tb.getcol('TIME')))
        Nbase=int(SHAP[0]/steps) # one write per time step
        for n in range(6,1,-1):
            if int(steps/n)==steps/n:
                Nbase *= n
                steps=int(steps/n)
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
    fg=tb.getcol('FLAG',nrow=Nbase,startrow=n*Nbase).reshape(-1)
    tsteps = time.time()-tic
    print(f'Read {Nbase} compressed complex visibilities from {DATACOL} column in {tsteps:.3f}s')
    s=vis.shape
    if True: #(COMPRESSOR == "mgard_complex")|(COMPRESSOR == "mgard"):
        vis=vis.reshape(-1)
        vis[np.where(fg==True)[0]]=0
        Inan=np.where(np.isnan(vis)==True)[0]
        vis[Inan]=0
        vis=vis.reshape(s)
    tic = time.time()
    if EXTERNAL:
        tb2.putcol('COPY_ADIOS',vis,nrow=Nbase,startrow=n*Nbase)
    else:
        tb.putcol('COPY_ADIOS',vis,nrow=Nbase,startrow=n*Nbase)
    tsteps = time.time()-tic
    s=np.nanstd(vis)
    print(f'Wrote {Nbase} compressed complex visibilities to ADIOS column with StdDev {s:.2e} in {tsteps:.3f}s')
  tsteps = time.time()-tot_tic
  if (steps>1): print(f'Total Read/Write compressed complex visibilities from {DATACOL}/COPY_ADIOS column in {tsteps:.3f}s')
  tb.close()
  if EXTERNAL: tb2.close()

  if EXTERNAL:
    tb2=table(TabName)
  tb=table(FILENAME,readonly=False)
  tot_tic = time.time()
  for n in range(steps):
    print('Read %d/%d\t'%(n,steps))
    #tic=time.time()
    #data=t.getcol('COPY_DATA',nrow=Nbase,startrow=n*Nbase)
    #tsteps = time.time()-tic
    #print(f'Read {Nbase} compressed complex visibilities from COPY_DATA column in {tsteps:.3f}s')
    tic=time.time()
    if EXTERNAL:
      vis=tb2.getcol('COPY_ADIOS',nrow=Nbase,startrow=n*Nbase)
    else:
      vis=tb.getcol('COPY_ADIOS',nrow=Nbase,startrow=n*Nbase)
    tsteps = time.time()-tic
    print(f'Read {Nbase} compressed complex visibilities from COPY_ADIOS column in {tsteps:.3f}s')
    if n==0:
        data=tb.getcol(DATACOL,nrow=Nbase,startrow=n*Nbase)
        a1=tb.getcol('ANTENNA1',nrow=Nbase,startrow=n*Nbase)
        a2=tb.getcol('ANTENNA2',nrow=Nbase,startrow=n*Nbase)
        I=np.where(a1!=a2)[0]
        tic=np.nanstd(data[I])
        data-=vis
        print('Data Difference:',np.nanmax(np.abs(data[I])),np.nanstd(data[I]),'StdDev',tic)
    tic = time.time()
    tb.putcol('COPY_DATA',vis,nrow=Nbase,startrow=n*Nbase)
    tsteps = time.time()-tic
    s=np.nanstd(vis[I])
    print(f'Wrote {Nbase} compressed complex visibilities of StdDev {s:.2e} to TILED column in {tsteps:.3f}s')
    if PLOT:
       mk_plot(vis-data,n)
  tsteps = time.time()-tot_tic
  if (steps>1): print(f'Total Read/Write compressed complex visibilities from COPY_ADIOS/COPY_DATA column in {tsteps:.3f}s')
  if EXTERNAL: 
      a_seq=tb2.getdminfo("COPY_ADIOS")["SEQNR"]
      tb2.close()
      a_on_disk_size = get_size(f'{TabName}/table.f{a_seq}.bp')
  else:
      a_seq=tb.getdminfo("COPY_ADIOS")["SEQNR"]
  t_seq=tb.getdminfo("COPY_DATA")["SEQNR"]
  tb.close()

  if EXTERNAL==False: a_on_disk_size = get_size(f'{FILENAME}/table.f{a_seq}.bp')
  n=f'{FILENAME}/table.f{t_seq}_TSM1'
  if (os.path.isfile(n)):
      t_on_disk_size = get_size(n)
      rat_on_disk_size=100.*a_on_disk_size/t_on_disk_size
      print(f'Native size: {t_on_disk_size} Compressed size: {a_on_disk_size} or {rat_on_disk_size:.1f}%')
  else:
      print(f'Compressed size: {a_on_disk_size}')



  if (EXTERNAL==False) & (DELETEOLD==True):
      tb=table(FILENAME,readonly=False)
      print('Remove old adios COPY')
      a_seq=tb.getdminfo("COPY_ADIOS")["SEQNR"]
      tb.removecols('COPY_ADIOS')
      print(f'Removing {FILENAME}/table.f{a_seq}.bp')
      shutil.rmtree(f'{FILENAME}/table.f{a_seq}.bp')
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
  
  HISTORY=True
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
  parser.add_argument("--external", action='store_true', help="(False) Write ADIOS externally")
  parser.add_argument("--plot", action='store_true', help="(False) Plot comparison histograms.")
  parser.add_argument("--reuse", action='store_true', help="(False) delete and remake columns")

  # various settings
  COMPRESSORS = ["mgard", "mgard_complex", "zfp", "sz", "None"];
  COMPRESSOR = "sz";
  MODE = 'ABS';
  ACCURACY = "0.4";
  FILENAMES = ['Raw/scienceData.G23_T0_A_01.SB32511.G23_T0_A_01.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_01.SB39697.G23_T0_A_01.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_02.SB39722.G23_T0_A_02.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_02.SB40329.G23_T0_A_02.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_03.SB40196.G23_T0_A_03.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_03.SB40479.G23_T0_A_03.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_04.SB40607.G23_T0_A_04.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_05.SB40712.G23_T0_A_05.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H01.SB65177.G23_T0_A_H01.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H02.SB65210.G23_T0_A_H02.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H03.SB65268.G23_T0_A_H03.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H04.SB65297.G23_T0_A_H04.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H05.SB65347.G23_T0_A_H05.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H06.SB65405.G23_T0_A_H06.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H07.SB65453.G23_T0_A_H07.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H08.SB65681.G23_T0_A_H08.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H09.SB65765.G23_T0_A_H09.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H10.SB66032.G23_T0_A_H10.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H11.SB66053.G23_T0_A_H11.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H12.SB66106.G23_T0_A_H12.beam17_SL.ms',
 'Raw/scienceData.G23_T0_A_H13.SB66134.G23_T0_A_H13.beam17_SL.ms',
 'Raw/scienceData.G23_T0_B_H06.SB65853.G23_T0_B_H06.beam17_SL.ms']
  FILENAME=FILENAMES[0]
  DATACOL = "DATA"
  DELETEOLD = True
  PLOT = False
  STEPS = -1
  args = parser.parse_args()
  if args.accuracy != ACCURACY:
      ACCURACY = args.accuracy
  if args.steps:
    STEPS = args.steps #(STEPS==False) # Swap the setting
  if args.external:
    EXTERNAL = (args.external==False) # Swap the setting for EXTERNAL
    print('Swapping EXTERNAL flag ',EXTERNAL)
  if args.reuse:
    DELETEOLD = (args.reuse==False) # Swap the setting for delete
    print('Swapping delete flag ',DELETEOLD)
  if args.plot:
    PLOT = True
  if args.filename != FILENAME: 
    if int(args.filename)<len(FILENAMES):
      FILENAME = FILENAMES[int(args.filename)]
    else: 
      print('FILENAME not known')
      exit()
  if args.datacol != DATACOL: 
    DATACOL = args.datacol
  if args.compressor not in COMPRESSORS:
      print(f"compressor argument needs to be one of {COMPRESSORS}")
      sys.exit()
  else:
      COMPRESSOR = args.compressor
  print(FILENAME,DATACOL,DELETEOLD,EXTERNAL,COMPRESSOR,ACCURACY)    
  run(DATACOL=DATACOL,FILENAME=FILENAME,EXTERNAL=EXTERNAL,
      STEPS=STEPS,ACCURACY=ACCURACY,COMPRESSOR=COMPRESSOR,
      DELETEOLD=DELETEOLD,PLOT=PLOT)
  sys.exit()
  
