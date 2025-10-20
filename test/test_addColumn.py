import argparse
import array
import json
import sys
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

from casacore.tables import (
  makearrcoldesc,
  makescacoldesc,
  table,
  maketabdesc,
  makedminfo)
import numpy as np
import adios2
from contextlib import contextmanager


# various settings
COMPRESSORS = ["mgard", "mgard_complex", "zfp", "sz"]
COMPRESSOR = "mgard"
COMPRESSOR1 = "mgard"
COMPRESSOR2 = "mgard"
ACCURACY = "0.1"
ACCURACY1 = "0.1"
ACCURACY2 = "0.1"
ORIG_SHAPE = [10000, 120, 4]
DIRNAME = "/scratch/ttable_adios_test"
PLOT = False
STEPS = False

def get_size(msdir):
    size = 0
    for root, dirs, files in os.walk(msdir):
        size += sum(os.path.getsize(os.path.join(root, name))
                    for name in files + dirs)
    return size

def create_column_desc(col_name:str, valuetype:str, cell_shape:list, compressor:str='', 
                              accuracy:str='', dm_group:str='group0', dm_name:str='asm1', 
                              tabdesc:dict={}, dm_type:str='Adios2StMan') -> tuple:
  """Create a column for a table using Adios2StMan or specified column manager.

  Args:
      col_name (str): Name of the column to be created.
      dm_type (str): Storage manager to be used (default Adios2StMan)
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
  if cell_shape is None or len(cell_shape) == 0:
    coldesc = makescacoldesc(col_name, '',
                valuetype=valuetype,
                datamanagergroup=dm_group, datamanagertype=dm_type
            )
  else:
    coldesc = makearrcoldesc(col_name, '',
                valuetype=valuetype, shape=cell_shape,
                datamanagergroup=dm_group, datamanagertype=dm_type
            )
  dminfo = {"TYPE": dm_type, "NAME": dm_name, "SPEC": {} }
  if dm_type == 'Adios2StMan' and compressor != ''  and accuracy != '':
     dminfo['SPEC'] = {"OPERATORPARAMS": {col_name: {"Operator": compressor, "Accuracy": accuracy}}}
  if not tabdesc:
      tabdesc = maketabdesc(coldesc)
      dminfo = makedminfo(
          tabdesc, 
          {dm_name: dminfo['SPEC']})

  return coldesc, dminfo, tabdesc

def write_ORIG_tiled():
  """Write generated visibilities to an ORIG column using the TiledStMan.
  """
  size = functools.reduce(operator.mul, ORIG_SHAPE[1:], ORIG_SHAPE[0] * 8)
  nrows = ORIG_SHAPE[0]
  cell_shape = ORIG_SHAPE[1:]

  rng = np.random.default_rng()
  vis = rng.normal(size= ORIG_SHAPE) + 1j*rng.normal(size=ORIG_SHAPE)
  vis = np.complex64(vis)

  print(f"Will write {size / 1024 / 1024:.2f} MB of data into {DIRNAME}\n\n")

  shape = vis.shape
  nrows = shape[0]
  cell_shape = shape[1:]
  dminfo={
          "TYPE": "TiledShapeStMan", "NAME": "tsm1", "SPEC": 
          {'MaxCacheSize': 0,
          'DEFAULTTILESHAPE': np.array([   4,    1, nrows/10], dtype=np.int32),
          'MAXIMUMCACHESIZE': 0,
          'HYPERCUBES': {'*1': {'CubeShape': np.array(ORIG_SHAPE, dtype=np.int32),
                                'TileShape': np.array([   4,    1, nrows/10], dtype=np.int32),
                                'CellShape': np.array([  4, 251], dtype=np.int32),
                                'BucketSize': 63936, 'ID': {}}},
          'SEQNR': 1,
          'IndexSize': 1}
        }

  coldesc = makearrcoldesc('ORIG', '',
              valuetype='complex', shape=cell_shape,
              datamanagergroup=dminfo['NAME'], datamanagertype='TiledShapeStMan'
          )

  tabdesc = maketabdesc(coldesc)
  dminfo = makedminfo(
      tabdesc, 
      {dminfo['NAME']: dminfo['SPEC']})
  tab = table(DIRNAME, tabledesc = tabdesc, dminfo = dminfo, readonly=False, ack=False)
  tic = time.time()
  tab.addrows(nrows)
  tab.putcol('ORIG', value = vis)
  tnocomp_complex = time.time() - tic
  print('wrote complex visibilities to ORIG column')
  return vis, tnocomp_complex

def write_ORIG_adios():
  """Write the ORIG column using the Adios2StMan.
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
  coldesc, dminfo, tabdesc = create_column_desc(
     'ORIG', 'complex', cell_shape, dm_group='asm1', dm_name='asm1')
  tab = table(DIRNAME, tabledesc=tabdesc, dminfo=dminfo, readonly=False, ack=False)
  tab.addrows(nrows)
  # write ORIG column
  tic = time.time()
  tab.putcol('ORIG',value=vis)
  tab.close()
  nocomp_complex = time.time() - tic
  return vis, nocomp_complex

def write_real_imag(vis:array) -> tuple:
  """
  Split a complex-valued visibility array into separate REAL and IMAG float columns
  and write them into an on-disk table using ADIOS2-backed column descriptors.

  Parameters
  ----------
  vis : array_like
    Complex-valued input array (e.g., numpy.ndarray) containing visibility data.
    The real part (np.real(vis)) will be written to a column named 'REAL' and
    the imaginary part (np.imag(vis)) will be written to a column named 'IMAG'.
  
  Returns
  -------
  tuple
    A tuple (comp_real, comp_imag) containing the elapsed wall-clock times in
    seconds for writing the 'REAL' and 'IMAG' columns, respectively.
  
  Side effects
  ------------
  - Opens or creates a table at DIRNAME (using table(..., readonly=False, ack=False)).
  - Adds two ADIOS2-backed float columns named 'REAL' and 'IMAG' to the table by
    calling create_column_desc(...) and tab.addcols(...).
  - Writes np.real(vis) into the 'REAL' column and np.imag(vis) into the 'IMAG'
    column using tab.putcol(...).
  - Uses compressor and accuracy settings provided by COMPRESSOR1/COMPRESSOR2 and
    ACCURACY1/ACCURACY2, and places data into data manager groups and names
    (dm_group='asm2'/'asm3', dm_name='asm2'/'asm3').
  - Updates and reuses the table descriptor (tabdesc) when adding the second column.

  Raises
  ------
  Exception
    Errors raised by the underlying table API, ADIOS2 descriptor creation, or
    IO operations (for example, if DIRNAME is not writable, the table cannot be
    created/modified, or vis has an incompatible type/shape).
    
  Notes
  -----
  - The function measures only the time taken to perform the column write
    operations (the intervals around tab.putcol calls) and returns those durations.
  - The implementation expects helper symbols and configuration (create_column_desc,
    table, DIRNAME, cell_shape, COMPRESSOR1, COMPRESSOR2, ACCURACY1, ACCURACY2, etc.)
    to be defined in the surrounding module scope.
  Example
  -------
  >>> comp_real, comp_imag = write_real_imag(vis)
  >>> print(f"REAL write time: {comp_real:.3f}s, IMAG write time: {comp_imag:.3f}s")
  """
  cell_shape = ORIG_SHAPE[1:]
  tab = table(DIRNAME, readonly=False, ack=False)
  # add and write REAL column
  coldesc, dminfo, tabdesc = create_column_desc(
     'REAL', 'float', cell_shape, dm_group='real', dm_name='real', 
     compressor=COMPRESSOR1, accuracy=ACCURACY1)
  tab.addcols(coldesc, dminfo)
  tic = time.time()
  tab.putcol('REAL',value=np.real(vis))
  comp_real = time.time() - tic

  # add and write IMAG column
  coldesc, dminfo, tabdesc = create_column_desc(
     'IMAG', 'float', cell_shape, dm_group='imag', dm_name='imag',
     compressor=COMPRESSOR2, accuracy=ACCURACY2, tabdesc=tabdesc)
  tab.addcols(coldesc, dminfo)
  tic = time.time()
  tab.putcol('IMAG',value=np.imag(vis))
  comp_imag = time.time() - tic
  print(tab.showstructure())
  tab.close()
  return comp_real, comp_imag

def make_DYSCO_column():
  """
  Just to keep the information how this would have to be done. DYSCO requires the table
  to have ANTENNA1, ANTENNA2, FIELD and DATA tables.
  """
  tab = table(DIRNAME, readonly=False, ack=False)
  coldesc = makearrcoldesc("DYSCO", np.complex64(0+0j), shape=ORIG_SHAPE[1:], options=1,
                           datamanagertype='DyscoStMan')
  coldesc['desc']['valueType']='complex' # bug in Dysco???
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
  cvis = None
  t = table(DIRNAME, ack=False)
  tic = time.time()
  vis = t.getcol('ORIG')
  tread_orig = time.time() - tic
  tic = time.time()
  cvis = t.getcol('DATA')
  tread_data = time.time() - tic
  tic = time.time()
  visr = t.getcol('REAL')
  tread_dreal = time.time() - tic
  tic = time.time()
  visi = t.getcol('IMAG')
  tread_dimag = time.time() - tic
  return vis, cvis, visr, visi, tread_orig, tread_data, tread_dreal, tread_dimag

def write_DATA_complex(vis:array):
  """Write (compressed) complex visibilities using ADIOS2StMan into DATA column."""
  tab = table(DIRNAME, readonly=False, ack=False)
  shape = vis.shape
  nrows = shape[0]
  cell_shape = shape[1:]
  coldesc, dminfo, tabdesc = create_column_desc(
    'DATA', 'complex', cell_shape, dm_group='asm2', dm_name='asm2', 
    compressor=COMPRESSOR, accuracy=ACCURACY)
  tab.addcols(coldesc, dminfo)
  tic = time.time()
  tab.putcol('DATA', value = vis)
  tdata_complex = time.time() - tic
  tab.close()
  print('wrote compressed complex visibilities to DATA column')
  return tdata_complex

def plot(vis:array, visr:array, visi:array, cvis:array=None):
  """Plot histograms of differences.

  Args:
      vis (array): Original visibilities.
      visr (array): REAL part of compressed visibilities.
      visi (array): IMAG part of compressed visibilities.
      cvis (array): If provided should contain the result of compressed complex visibilities.
  """
  print(f"Plotting (close the plot window to exit script)...")
  if cvis is None:
    plt.hist(np.real(vis).reshape(-1)-visr.reshape(-1), 
             label=f'real [{COMPRESSOR1}:{ACCURACY1}]',
             align='mid')
    plt.hist(np.imag(vis).reshape(-1)-visi.reshape(-1),
             label=f'imag [{COMPRESSOR2}:{ACCURACY2}]',
             align='mid', alpha=0.5)
  else:
    if visr is not None and visi is not None:

      rcvis = np.complex64(visr + 1j*visi)
      diff = np.absolute(vis.reshape(-1)) - np.absolute(rcvis.reshape(-1))
      xscale = max(diff)
      plt.hist(diff,
              label=f'orig-(real+1j*imag)(compressed) [{COMPRESSOR1}:{ACCURACY1}]',
              align='mid', bins = 100, hatch='/')
    plt.rcParams.update({'hatch.color': 'black'})

    diff = np.real(vis.reshape(-1))-visr.reshape(-1)
    xscale = max(xscale, max(diff)) * 1.5
    plt.hist(diff,
              label=f'real(orig)-real(compressed) [{COMPRESSOR1}:{ACCURACY1}]',
              align='mid', alpha=0.5, bins = 100, hatch = '|')
    plt.rcParams.update({'hatch.color': 'red'})

    diff = np.absolute(vis.reshape(-1)) - np.absolute(cvis.reshape(-1))
    xscale = max(xscale, max(diff))
    plt.hist(diff,
              label=f'orig-compressed [{COMPRESSOR}:{ACCURACY}]',
              align='mid', alpha=0.5, bins = 100, hatch='-')
    plt.rcParams.update({'hatch.color': 'blue'})

    acc = float(ACCURACY)
    plt.plot([-acc, -acc],plt.ylim(), color='white')
    plt.plot([acc, acc],plt.ylim(), color='white')
    plt.xlim((-xscale, xscale))
       
  plt.yscale("log")
  plt.title(f"original-compressed ({COMPRESSOR})")
  plt.xlabel(f"original-compressed")
  plt.ylabel("Log(Number) of occurance")
  plt.legend()
  plt.show()

def write_steps(vis:array, col:str='STEPS', dm_type:str='Adios2StMan'):
  """Write data in steps into a new STEPS table.
  """
  steps = 3
  tab = table(DIRNAME, readonly=False, ack=False)
  shape = vis.shape
  nrows = shape[0]
  nrows_step = nrows//steps
  cell_shape = shape[1:] if len(shape) > 1 else None
  coldesc, dminfo, tabdesc = create_column_desc(
    col, 'complex', cell_shape, dm_type=dm_type,
    dm_group=col.lower(), dm_name=col.lower(), 
    compressor='', accuracy='')
  tab.addcols(coldesc, dminfo)
  tsteps = 0
  for step in range(steps):
    value = vis[step*nrows_step:(step+1)*nrows_step]
    tic = time.time()
    tab.putcol(col, 
               value=value,
               startrow = step*nrows_step,
               nrow = nrows_step
               )
    tsteps +=time.time() - tic
    print(f'{nrows_step} rows with {len(np.where(value.reshape(-1)==0)[0])} zero values written in {tsteps}s')
  tab.close()
  print(f'wrote compressed complex visibilities to {col} column')
  return tsteps

def read_steps(col:str='STEPS', steps:int=3):
  """
  Read a column from a table in multiple reads and return the assembled array.

  This function opens a table at DIRNAME (readonly), determines the per-row data shape
  from the column descriptor, allocates an output NumPy array of the given dtype, and
  reads the table column in `steps` sequential chunks using table.getcol. Timing for
  each chunk is printed and the total read time is printed before returning. The table
  is closed before returning.

  Parameters
  ----------
  col : str, optional
    Name of the column to read from the table. Default: 'STEPS'.
  steps : int, optional
    Number of equally-sized steps (chunks) to split the read into. Default: 3.
    Each step reads floor(nrows / steps) rows. If `steps <= 0` a ValueError is raised.

  Returns
  -------
  numpy.ndarray
    An array of shape (nrows_used, shape[0]) where `shape[0]` is taken from the
    column descriptor for `col`, and `nrows_used` is `steps * floor(nrows / steps)`.
    The array dtype is `valuetype`.

  Notes
  -----
  - The function uses integer division to compute the per-step row count:
    nrows_step = nrows // steps. Any remainder rows (nrows % steps) are not read
    by this implementation and therefore are not present in the returned array.
  - The function depends on a global DIRNAME value and on a `table` API that
    provides `nrows()`, `getcoldesc(col)` returning a dict with key 'shape',
    `getcol(col, startrow, nrow)`, and `close()`.
  - The function prints timing information for each step and a final summary.
  - The table is opened with readonly=True and ack=False.

  Raises
  ------
  ValueError
    If `steps` is not a positive integer or if resulting `nrows_step` is zero
    (i.e., when `steps` > nrows).
  KeyError, IndexError, TypeError
    May be raised if the table API does not provide the expected descriptors or
    if the column shape metadata is not in the expected format.

  Example
  -------
  >>> # assuming DIRNAME and table API are available and tab has nrows = 300
  >>> arr = read_steps(col='DATA', valuetype=np.complex128, steps=3)
  >>> arr.shape
  (300, N)  # N is the per-row length obtained from the column descriptor
  """
  tab = table(DIRNAME, readonly=True, ack=False)
  nrows = tab.nrows()
  nrows_step = nrows//steps
  tsteps = 0
  for step in range(steps):
    tic = time.time()
    startrow = step*nrows_step
    v = tab.getcol(col,
               startrow = startrow,
               nrow = nrows_step
               )
    tsteps +=time.time() - tic
    if step == 0:
      value = np.array(v)
    else:
      value = np.append(value, np.array(v), axis = 0)
    print(f'{nrows_step} rows read in {tsteps:.3f}s')
  tab.close()
  print(f'read compressed complex visibilities to {col} column in {tsteps:.2f}s')
  return value


def run()-> tuple:
  """Write and read the table and create a DATA column from the compressed
  REAL and IMAG parts. Plot differences histograms.
  """
  size = functools.reduce(operator.mul, ORIG_SHAPE[1:], ORIG_SHAPE[0] * 8)

  print("Settings:")
  print(f"  Compressor for DATA column: {COMPRESSOR} (Accuracy: {ACCURACY})")
  print(f"  Compressor for REAL column: {COMPRESSOR1} (Accuracy: {ACCURACY1})")
  print(f"  Compressor for IMAG column: {COMPRESSOR2} (Accuracy: {ACCURACY2})")
  print(f"  Data shape: {ORIG_SHAPE}")
  print(f"  Output directory: {DIRNAME}")
  print()

  vis, tnocomp_complex = write_ORIG_tiled()
  tcomp_real, tcomp_imag = write_real_imag(vis)
  twrite_complex = write_DATA_complex(vis)

  vis, cvis, visr, visi, tread_complex, read_dcomplex, tdecomp_real, tdecomp_imag = read_table()

  r_on_disk_size = get_size(f'{DIRNAME}/table.f1.bp')
  print(f'ORIG write time: {tnocomp_complex:.3f}')
  print(f'REAL[{COMPRESSOR1}] compression and write time: {tcomp_real:.3f}')
  i_on_disk_size = get_size(f'{DIRNAME}/table.f2.bp')
  print(f'IMAG[{COMPRESSOR2}] compression and write time: {tcomp_imag:.3f}\n')
  print('Total compression and write time: '
        f'{(tcomp_real+tcomp_imag):.3f} ({((tcomp_real+tcomp_imag)/tnocomp_complex):.1f}x)\n\n')

  print(f'ORIG read time: {(tread_complex):.3f} s')
  print(f'REAL[{COMPRESSOR1}] decompression and read time: {(tdecomp_real):.3f} s')
  print(f'REAL compression ratio: {size / r_on_disk_size:.2f}')
  print(f'IMAG[{COMPRESSOR2}] decompression and read time: {(tdecomp_imag):.3f} s')
  print(f'IMAG compression ratio: {size / i_on_disk_size:.2f}\n')
  print('Total decompression and read time: '
        f'{(tdecomp_real+tdecomp_imag):.3f} s ({((tdecomp_real+tdecomp_imag)/tread_complex):.1f}x)\n\n')
  if PLOT:
    plot(vis, visr, visi, cvis)
  if STEPS:
    write_steps(vis)
  return vis, visr, visi, cvis

if __name__ == "__main__":  
  parser = argparse.ArgumentParser(description=
                                   'Test the column-wise compression using the Adios2StMan storage manager in casacore tables')
  parser.add_argument("--compressor", type=str, default=COMPRESSOR, help="Global data compressor")
  parser.add_argument("--compressor1", type=str, default=COMPRESSOR1, help="Compressor for REAL column")
  parser.add_argument("--compressor2", type=str, default=COMPRESSOR2, help="Compressor for IMAG column")
  parser.add_argument("--accuracy", type=str, default=ACCURACY, help="Global accuracy for data columns")
  parser.add_argument("--accuracy1", type=str, default=ACCURACY1, help="Accuracy for REAL column compressor")
  parser.add_argument("--accuracy2", type=str, default=ACCURACY2, help="Accuracy for IMAG column compressor")
  parser.add_argument("--shape", type=str, default=ORIG_SHAPE, help="Shape of the data array")
  parser.add_argument("--dirname", type=str, default=DIRNAME, help="Output filename")
  parser.add_argument("--steps", action='store_true', help="(False) Write a STEPS column in steps.")
  parser.add_argument("--plot", action='store_true', help="(False) Plot comparison histograms.")

  args = parser.parse_args()
  if args.accuracy != ACCURACY:
      ACCURACY = ACCURACY1 = ACCURACY2 = args.accuracy
  elif args.accuracy1 != ACCURACY1 or args.accuracy2 != ACCURACY2:
    ACCURACY1 = args.accuracy1
    ACCURACY2 = args.accuracy2
  if args.shape != ORIG_SHAPE:
    ORIG_SHAPE = json.loads(args.shape)
  if args.steps:
    STEPS = True
  if args.plot:
    PLOT = True
  if args.dirname != DIRNAME: 
    DIRNAME = args.dirname
  if args.compressor != COMPRESSOR:
    if args.compressor not in COMPRESSORS:
      print(f"compressor argument needs to be one of {COMPRESSORS}")
      sys.exit()
    if args.compressor == 'mgard_complex':
      COMPRESSOR = 'mgard_complex'
      COMPRESSOR1 = COMPRESSOR2 = 'mgard'
    else:
      COMPRESSOR = COMPRESSOR1 = COMPRESSOR2 = args.compressor
    vis, visr, visi, cvis = run()
    sys.exit()
  elif args.compressor1 != COMPRESSOR1 or args.compressor2 != COMPRESSOR2:
    COMPRESSOR1 = args.compressor1
    COMPRESSOR2 = args.compressor2
  vis, visr, visi, cvis = run()
