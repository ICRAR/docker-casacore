import argparse
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

def run():
  print("Settings:")
  print(f"  Compressor for REAL column: {COMPRESSOR1} (Accuracy: {ACCURACY1})")
  print(f"  Compressor for IMAG column: {COMPRESSOR2} (Accuracy: {ACCURACY2})")
  print(f"  Data shape: {ORIG_SHAPE}")
  print(f"  Output directory: {DIRNAME}")
  print()
  # produce some data
  nrows = ORIG_SHAPE[0]
  cell_shape = ORIG_SHAPE[1:]

  rng = np.random.default_rng()
  vis = rng.normal(size= ORIG_SHAPE) + 1j*rng.normal(size=ORIG_SHAPE)
  vis = np.complex64(vis)
  size = functools.reduce(operator.mul, cell_shape, nrows * 8)

  print(f"Will write {size / 1024 / 1024:.2f} MB of data into {DIRNAME}\n\n")

  # setup table
  tabdesc = maketabdesc((
          makearrcoldesc('REAL', '',
              valuetype='float', shape=cell_shape,
              datamanagergroup='group0', datamanagertype='Adios2StMan'
          ),
          makearrcoldesc('ORIG', '',
              valuetype='complex', shape=cell_shape,
              datamanagergroup='group1', datamanagertype='Adios2StMan'
          )
          )
      )

  # setup dminfo with compression
  dminfo = makedminfo(
          tabdesc,
          {
              'group0': {
                  'OPERATORPARAMS': {
                      'REAL': {
                          'Operator': COMPRESSOR1,
                          'Accuracy': str(ACCURACY1),
                      }
                  }
              }
          }
      )

  # Create the table
  t = table(DIRNAME, tabledesc=tabdesc, dminfo=dminfo, ack=False)

  # add as many rows as there are rows in the data
  t.addrows(nrows)

  # write the columns and close the table
  tic = time.time()
  t.putcol('ORIG',value=vis)
  nocomp_complex = time.time() - tic
  tic = time.time()
  t.putcol('REAL',value=np.real(vis))
  comp_real = time.time() - tic
  cmi = t.getdminfo()
  t.addcols(makearrcoldesc("IMAG", 0., shape=cell_shape), 
            dminfo={
                "TYPE": "Adios2StMan", "NAME": "asm2", "SPEC": 
                {"OPERATORPARAMS": {"IMAG": {"Operator": COMPRESSOR2, "Accuracy": ACCURACY2}}}
                }
            )
  tic = time.time()
  t.putcol('IMAG',value=np.imag(vis))
  comp_imag = time.time() - tic

  print(t.showstructure())
  t.close()

  on_disk_size = get_size(f'{DIRNAME}/table.f0.bp')
  # print(f'REAL size raw {COMPRESSOR1}: {size / 1024 / 1024:.2f} MB')
  # print(f'Table size on disk: {on_disk_size / 1024 / 1024:.2f} MB')
  print(f'ORIG write time: {nocomp_complex:.3f}')
  print(f'REAL compression and write time: {comp_real:.3f}')
  on_disk_size = get_size(f'{DIRNAME}/table.f1.bp')
  # print(f'IMAG size raw {COMPRESSOR2}: {size / 1024 / 1024:.2f} MB')
  # print(f'Table size on disk: {on_disk_size / 1024 / 1024:.2f} MB')
  print(f'IMAG compression and write time: {comp_imag:.3f}\n')
  print(f'Total compression and write time: {(comp_real+comp_imag):.3f}\n\n')

  # Check the difference between original and compressed
  # af1 = adios2.FileReader('ttable_adios_test/table.f0.bp/')
  # visr = af1.read('REAL',start=[0,0,0],count=ORIG_SHAPE)
  # af2 = adios2.FileReader('ttable_adios_test/table.f1.bp/')
  # visi = af2.read('IMAG',start=[0,0,0],count=ORIG_SHAPE)
  t = table(DIRNAME, readonly=False, ack=False)
  tic = time.time()
  vis = t.getcol('ORIG')
  read_complex = time.time() - tic
  tic = time.time()
  visr = t.getcol('REAL')
  decomp_real = time.time() - tic
  tic = time.time()
  visi = t.getcol('IMAG')
  decomp_imag = time.time() - tic
  print(f'ORIG read time: {(read_complex):.3f} s')
  print(f'REAL decompression and read time: {(decomp_real):.3f} s')
  print(f'REAL compression ratio: {size / on_disk_size:.2f}')
  print(f'IMAG decompression and read time: {(decomp_imag):.3f} s')
  print(f'IMAG compression ratio: {size / on_disk_size:.2f}\n')
  print(f'Total decompression and read time: {(decomp_real+decomp_imag):.3f} s\n\n')

  t.addcols(makearrcoldesc("DATA", 0.+0j, shape=cell_shape), 
            dminfo={
                "TYPE": "TiledShapeStMan", "NAME": "tsm1", "SPEC": 
                {'MaxCacheSize': 0,
                 'DEFAULTTILESHAPE': np.array([   4,    1, 1998], dtype=np.int32),
                 'MAXIMUMCACHESIZE': 0,
                 'HYPERCUBES': {'*1': {'CubeShape': np.array([      4,     251, 1928070], dtype=np.int32),
                                       'TileShape': np.array([   4,    1, 1998], dtype=np.int32),
                                       'CellShape': np.array([  4, 251], dtype=np.int32),
                                       'BucketSize': 63936, 'ID': {}}},
  'SEQNR': 2,
  'IndexSize': 1}
                }
            )
  cvis = np.complex64(visr + 1j*visi) # cast back to complex
  t.putcol('DATA', value = cvis)
  print('wrote decompressed complex visibilities to DATA column')

  return vis, visr, visi, cvis

def plot(vis, visr, visi):
  # Plot the difference
  print(f"Plotting (close the plot window to exit script)...")
  plt.hist(np.real(vis).reshape(-1)-visr.reshape(-1), label=f'real [{COMPRESSOR1}:{ACCURACY1}]')
  plt.hist(np.imag(vis).reshape(-1)-visi.reshape(-1), label=f'imag [{COMPRESSOR2}:{ACCURACY2}]')
  plt.yscale("log")
  plt.title(f"original-compressed")
  plt.xlabel(f"original-compressed")
  plt.ylabel("Log(Number) of occurance")
  plt.legend()
  plt.show()

def main():
  vis, visr, visi, cvis = run()
  plot(vis, visr, visi)

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
  main()
