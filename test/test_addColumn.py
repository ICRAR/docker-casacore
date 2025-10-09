import argparse
import array
from turtle import shape
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
  python test_addColumn.py [--compressor COMP] [--compressor1 COMP1] [--compressor2 COMP2]
               [--accuracy ACC] [--accuracy1 ACC1] [--accuracy2 ACC2]
               [--shape N M K] [--dirname DIR]
"""
import functools
import operator
import os, time
from matplotlib import pyplot as plt

from casacore.tables import (makearrcoldesc, table,
                          maketabdesc, makedminfo)
import numpy as np
import adios2
from contextlib import contextmanager


# various settings
COMPRESSOR = "mgard"
COMPRESSOR1 = "mgard"
COMPRESSOR2 = "mgard"
ACCURACY = "0.1"
ACCURACY1 = "0.1"
ACCURACY2 = "0.1"
ORIG_SHAPE = [10000, 120, 4]
DIRNAME = "ttable_adios_test"

def get_size(msdir):
    size = 0
    for root, dirs, files in os.walk(msdir):
        size += sum(os.path.getsize(os.path.join(root, name))
                    for name in files + dirs)
    return size

def create_ADIOS2_column_desc(col_name:str, valuetype:str, cell_shape:list, compressor:str='', 
                              accuracy:str='', dm_group:str='group0', dm_name:str='asm1', tabdesc:dict={}) -> tuple:
  """Create an ADIOS2 column for a table.

  Args:
      col_name (str): Name of the column to be created.
      valuetype (str): The type of the value of the column.
      cell_shape (list): Shape of a single cell.
      compressor (str, optional): Compressor operator to be used. Defaults to 'mgard'.
      accuracy (str, optional): Accuracy for the compressor. Defaults to ACCURACY.
      dm_group (str, optional): DataManager group name. Defaults to 'group0'.
      dm_name (str, optional): DataManager name. Defaults to 'asm1'.
      tabdesc (dict, optional): Tablesecription dict. Defaults to {}.

  Returns:
      tuple: (coldesc, dminfo, tabdesc)
  """
  coldesc = makearrcoldesc(col_name, '',
              valuetype=valuetype, shape=cell_shape,
              datamanagergroup=dm_group, datamanagertype='Adios2StMan'
          )
  if not tabdesc:
      tabdesc = maketabdesc(coldesc)
      dminfo = makedminfo(
          tabdesc, 
          {dm_name: {"OPERATORPARAMS": {col_name: {"Operator": compressor, "Accuracy": accuracy}}}})
  elif compressor and accuracy:
     dminfo = {
                "TYPE": "Adios2StMan", "NAME": dm_name, "SPEC": 
                {"OPERATORPARAMS": {col_name: {"Operator": compressor, "Accuracy": accuracy}}}        
     }
  else:
     dminfo = {"TYPE": "Adios2StMan", "NAME": dm_name, "SPEC": {} }

  return coldesc, dminfo, tabdesc

def write_table() -> tuple[float, float, float]:
  """Create table and write initial data columns to it.

  Args:
      size (int): _description_

  Returns:
      tuple[float, float, float]: _description_
  """
  size = functools.reduce(operator.mul, ORIG_SHAPE[1:], ORIG_SHAPE[0] * 8)
  nrows = ORIG_SHAPE[0]
  cell_shape = ORIG_SHAPE[1:]

  rng = np.random.default_rng()
  vis = rng.normal(size= ORIG_SHAPE) + 1j*rng.normal(size=ORIG_SHAPE)
  vis = np.complex64(vis)

  print(f"Will write {size / 1024 / 1024:.2f} MB of data into {DIRNAME}\n\n")

  # setup table
  # Create the table
  coldesc, dminfo, tabdesc = create_ADIOS2_column_desc(
     'ORIG', 'complex', cell_shape, dm_group='asm1', dm_name='asm1')
  t = table(DIRNAME, tabledesc=tabdesc, dminfo=dminfo, ack=False)
  t.addrows(nrows)

  # write ORIG column
  tic = time.time()
  t.putcol('ORIG',value=vis)
  nocomp_complex = time.time() - tic

  # add and write REAL column
  coldesc, dminfo, tabdesc = create_ADIOS2_column_desc(
     'REAL', 'float', cell_shape, dm_group='asm2', dm_name='asm2', 
     compressor=COMPRESSOR1, accuracy=ACCURACY1, tabdesc=tabdesc)
  t.addcols(coldesc, dminfo)
  tic = time.time()
  t.putcol('REAL',value=np.real(vis))
  comp_real = time.time() - tic

  # add and write IMAG column
  coldesc, dminfo, tabdesc = create_ADIOS2_column_desc(
     'IMAG', 'float', cell_shape, dm_group='asm3', dm_name='asm3',
     compressor=COMPRESSOR2, accuracy=ACCURACY2, tabdesc=tabdesc)
  t.addcols(coldesc, dminfo)
  tic = time.time()
  t.putcol('IMAG',value=np.imag(vis))
  comp_imag = time.time() - tic

  print(t.showstructure())
  t.close()
  return nocomp_complex, comp_real, comp_imag

def make_DYSCO_column():
  """
  Just to keep the information how this would have to be done. DYSCO requires the table
  to have ANTENNA1, ANTENNA2, FIELD and DATA tables.
  """
  tab = table(DIRNAME, readonly=False, ack=False)
  coldesc = makearrcoldesc("DYSCO", np.complex64(0+0j), shape=ORIG_SHAPE[1:], options=1,
                           datamanagertype='DyscoStMan')
  coldesc['desc']['valueType']='complex' # bug in Dysco???
  print(coldesc)
  tab.addcols(coldesc,
            dminfo={
                    "TYPE": "DyscoStMan", "NAME": "dysco", "SPEC": {
                        'dataBitCount': 8,
                        'weightBitCount': 12,
                        'distribution': 'TruncatedGaussian',
                        'normalization': 'AF',
                        'studentTNu': 0.0,
                        'distributionTruncation': 2.5
                                 }
                    }
                )
  return

def read_table():
  """Read the table back

  Returns:
      tuple: _description_
  """
  t = table(DIRNAME, ack=False)
  tic = time.time()
  vis = t.getcol('ORIG')
  read_complex = time.time() - tic
  tic = time.time()
  visr = t.getcol('REAL')
  decomp_real = time.time() - tic
  tic = time.time()
  visi = t.getcol('IMAG')
  decomp_imag = time.time() - tic

  return vis, visr, visi, read_complex, decomp_real, decomp_imag

def write_DATA(visr:array, visi:array):
  """Write a DATA column with complex visibilites

  Args:
      tab (table): The table created by write_table
      visr (array): REAL part of the visibilities
      visi (array): IMAG part of the visibilities
  """
  tab = table(DIRNAME, readonly=False, ack=False)
  shape = visr.shape
  nrows = shape[0]
  cell_shape = shape[1:]
  tab.addcols(makearrcoldesc("DATA", 0.+0j, shape=cell_shape), 
            dminfo={
                    "TYPE": "TiledShapeStMan", "NAME": "tsm1", "SPEC": 
                    {'MaxCacheSize': 0,
                    'DEFAULTTILESHAPE': np.array([   4,    1, nrows/10], dtype=np.int32),
                    'MAXIMUMCACHESIZE': 0,
                    'HYPERCUBES': {'*1': {'CubeShape': np.array(ORIG_SHAPE, dtype=np.int32),
                                          'TileShape': np.array([   4,    1, nrows/10], dtype=np.int32),
                                          'CellShape': np.array([  4, 251], dtype=np.int32),
                                          'BucketSize': 63936, 'ID': {}}},
                    'SEQNR': 2,
                    'IndexSize': 1}
                  }
            )
  cvis = np.complex64(visr + 1j*visi) # cast back to complex
  tab.putcol('DATA', value = cvis)
  print('wrote decompressed complex visibilities to DATA column')

def plot(vis:array, visr:array, visi:array):
  """Plot histograms of differences.

  Args:
      vis (array): Original visibilities
      visr (array): REAL part of visibilities
      visi (array): IMAG part of visibilities
  """
  print(f"Plotting (close the plot window to exit script)...")
  plt.hist(np.real(vis).reshape(-1)-visr.reshape(-1), label=f'real [{COMPRESSOR1}:{ACCURACY1}]')
  plt.hist(np.imag(vis).reshape(-1)-visi.reshape(-1), label=f'imag [{COMPRESSOR2}:{ACCURACY2}]')
  plt.yscale("log")
  plt.title(f"original-compressed")
  plt.xlabel(f"original-compressed")
  plt.ylabel("Log(Number) of occurance")
  plt.legend()
  plt.show()

def run()-> tuple:
  """Write and read the table and create a DATA column from the compressed
  REAL and IMAG parts. Plot differences histograms.
  """
  size = functools.reduce(operator.mul, ORIG_SHAPE[1:], ORIG_SHAPE[0] * 8)

  print("Settings:")
  print(f"  Compressor for REAL column: {COMPRESSOR1} (Accuracy: {ACCURACY1})")
  print(f"  Compressor for IMAG column: {COMPRESSOR2} (Accuracy: {ACCURACY2})")
  print(f"  Data shape: {ORIG_SHAPE}")
  print(f"  Output directory: {DIRNAME}")
  print()

  nocomp_complex, comp_real, comp_imag = write_table()
  vis, visr, visi, read_complex, decomp_real, decomp_imag = read_table()

  r_on_disk_size = get_size(f'{DIRNAME}/table.f1.bp')
  print(f'ORIG write time: {nocomp_complex:.3f}')
  print(f'REAL compression and write time: {comp_real:.3f}')
  i_on_disk_size = get_size(f'{DIRNAME}/table.f2.bp')
  print(f'IMAG compression and write time: {comp_imag:.3f}\n')
  print(f'Total compression and write time: {(comp_real+comp_imag):.3f}\n\n')

  print(f'ORIG read time: {(read_complex):.3f} s')
  print(f'REAL[{COMPRESSOR1}] decompression and read time: {(decomp_real):.3f} s')
  print(f'REAL compression ratio: {size / r_on_disk_size:.2f}')
  print(f'IMAG[{COMPRESSOR2}] decompression and read time: {(decomp_imag):.3f} s')
  print(f'IMAG compression ratio: {size / i_on_disk_size:.2f}\n')
  print(f'Total decompression and read time: {(decomp_real+decomp_imag):.3f} s\n\n')

  vis, visr, visi, read_complex, decomp_real, decomp_imag = read_table()

  write_DATA(visr, visi)
  plot(vis, visr, visi)
  return

if __name__ == "__main__":  
  parser = argparse.ArgumentParser(description=
                                   'Test the column-wise compression using the Adios2StMan storage manager in casacore tables')
  parser.add_argument("--compressor", type=str, default=COMPRESSOR, help="Compressor for REAL and IMAG columns")
  parser.add_argument("--compressor1", type=str, default=COMPRESSOR1, help="Compressor for REAL column")
  parser.add_argument("--compressor2", type=str, default=COMPRESSOR2, help="Compressor for IMAG column")
  parser.add_argument("--accuracy", type=str, default=ACCURACY, help="Accuracy for REAL and IMAG columns")
  parser.add_argument("--accuracy1", type=str, default=ACCURACY1, help="Accuracy for REAL column compressor")
  parser.add_argument("--accuracy2", type=str, default=ACCURACY2, help="Accuracy for IMAG column compressor")
  parser.add_argument("--shape", type=int, nargs=3, default=ORIG_SHAPE, help="Shape of the data array")
  parser.add_argument("--dirname", type=str, default=DIRNAME, help="Output filename")

  args = parser.parse_args()
  print(args)
  if args.compressor != COMPRESSOR:
     COMPRESSOR = COMPRESSOR1 = COMPRESSOR2 = args.compressor
  elif args.compressor1 != COMPRESSOR1 or args.compressor2 != COMPRESSOR2:
    COMPRESSOR1 = args.compressor1
    COMPRESSOR2 = args.compressor2
  if args.accuracy != ACCURACY:
      ACCURACY = ACCURACY1 = ACCURACY2 = args.accuracy
  elif args.accuracy1 != ACCURACY1 or args.accuracy2 != ACCURACY2:
    ACCURACY1 = args.accuracy1
    ACCURACY2 = args.accuracy2
  if hasattr(args, 'shape'):
    ORIG_SHAPE = args.shape
  if hasattr(args, 'DIRNAME'): 
    DIRNAME = args.filename
  run()
