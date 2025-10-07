import argparse
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
DATA_SHAPE = [10000, 120, 4]
DIRNAME = "ttable_adios_test"

def get_size(msdir):
    size = 0
    for root, dirs, files in os.walk(msdir):
        size += sum(os.path.getsize(os.path.join(root, name))
                    for name in files + dirs)
    return size

def main():
  print("Settings:")
  print(f"  Compressor for REAL column: {COMPRESSOR1} (Accuracy: {ACCURACY1})")
  print(f"  Compressor for IMAG column: {COMPRESSOR2} (Accuracy: {ACCURACY2})")
  print(f"  Data shape: {DATA_SHAPE}")
  print(f"  Output directory: {DIRNAME}")
  print()
  # produce some data
  nrows = DATA_SHAPE[0]
  cell_shape = DATA_SHAPE[1:]

  rng = np.random.default_rng()
  vis = rng.normal(size= DATA_SHAPE) + 1j*rng.normal(size=DATA_SHAPE)
  vis = np.complex64(vis)
  size = functools.reduce(operator.mul, cell_shape, nrows * 8)

  print(f"Will write {size / 1024 / 1024:.2f} MB of data into {DIRNAME}\n\n")

  # setup table
  tabdesc = maketabdesc(
          makearrcoldesc('REAL', '',
              valuetype='float', shape=cell_shape,
              datamanagergroup='group0', datamanagertype='Adios2StMan'
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
  t.putcol('REAL',value=np.real(vis))
  comp_real = time.time() - tic
  cmi = t.getdminfo()
  t.addcols(makearrcoldesc("IMAG", 0., shape=vis.shape[1:]), 
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
  print(f'REAL compression and write time: {comp_real:.3f}')
  on_disk_size = get_size(f'{DIRNAME}/table.f1.bp')
  # print(f'IMAG size raw {COMPRESSOR2}: {size / 1024 / 1024:.2f} MB')
  # print(f'Table size on disk: {on_disk_size / 1024 / 1024:.2f} MB')
  print(f'IMAG compression and write time: {comp_imag:.3f}\n')
  print(f'Total compression and write time: {(comp_real+comp_imag):.3f}\n\n')

  # Check the difference between original and compressed
  # af1 = adios2.FileReader('ttable_adios_test/table.f0.bp/')
  # visr = af1.read('REAL',start=[0,0,0],count=DATA_SHAPE)
  # af2 = adios2.FileReader('ttable_adios_test/table.f1.bp/')
  # visi = af2.read('IMAG',start=[0,0,0],count=DATA_SHAPE)
  t = table(DIRNAME, readonly=True, ack=False)
  tic = time.time()
  visr = t.getcol('REAL')
  decomp_real = time.time() - tic
  tic = time.time()
  visi = t.getcol('IMAG')
  decomp_imag = time.time() - tic
  print(f'REAL decompression and read time: {(decomp_real):.3f} s')
  print(f'REAL compression ratio: {size / on_disk_size:.2f}')
  print(f'IMAG decompression and read time: {(decomp_imag):.3f} s')
  print(f'IMAG compression ratio: {size / on_disk_size:.2f}\n')
  print(f'Total decompression and read time: {(decomp_real+decomp_imag):.3f} s\n\n')
  cvis = np.complex64(visr + 1j*visi) # cast back to complex

  return vis, visr, visi

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

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Test Adios2StMan column compression.")
  parser.add_argument("--compressor", type=str, default=COMPRESSOR, help="Compressor for REAL and IMAG columns")
  parser.add_argument("--compressor1", type=str, default=COMPRESSOR1, help="Compressor for REAL column")
  parser.add_argument("--compressor2", type=str, default=COMPRESSOR2, help="Compressor for IMAG column")
  parser.add_argument("--accuracy", type=str, default=ACCURACY, help="Accuracy for REAL and IMAG columns")
  parser.add_argument("--accuracy1", type=str, default=ACCURACY1, help="Accuracy for REAL column compressor")
  parser.add_argument("--accuracy2", type=str, default=ACCURACY2, help="Accuracy for IMAG column compressor")
  parser.add_argument("--shape", type=int, nargs=3, default=DATA_SHAPE, help="Shape of the data array")
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
    DATA_SHAPE = args.shape
  if hasattr(args, 'DIRNAME'): 
    DIRNAME = args.filename
  vis, visr, visi = main()
  plot(vis, visr, visi)