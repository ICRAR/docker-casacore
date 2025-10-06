import functools
import operator
import os
from matplotlib import pyplot as plt

from casacore.tables import (makearrcoldesc, table,
                          maketabdesc, makedminfo)
import numpy as np
import adios2

def get_size(msdir):
    size = 0
    for root, dirs, files in os.walk(msdir):
        size += sum(os.path.getsize(os.path.join(root, name))
                    for name in files + dirs)
    return size

# produce some data
rng = np.random.default_rng()
nrows = 10000
vis = rng.random(dtype=np.float32, size=(nrows, 120, 4)) + 1j*rng.random(dtype=np.float32, size=(nrows, 120, 4))
vis = vis.view(np.complex64)
cell_shape = vis.shape[1:]
size = functools.reduce(operator.mul, cell_shape, nrows * 8)

# various settings
compressor = "mgard"
accuracy = "0.1"
filename = "ttable_adios_test"
print(f"Will write {size / 1024 / 1024:.2f} MB of data into {filename}\n\n")

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
                        'Operator': compressor,
                        'Accuracy': str(accuracy),
                    }
                }
            }
        }
    )

# Create the table
t = table(filename, tabledesc=tabdesc, dminfo=dminfo, ack=False)

# add as many rows as there are rows in the data
t.addrows(nrows)

# write the column data and close the table
t.putcol('REAL',value=np.real(vis))
cmi = t.getdminfo()
t.addcols(makearrcoldesc("IMAG", 0., shape=vis.shape[1:]), dminfo={"TYPE": "Adios2StMan", "NAME": "asm2", "SPEC": {"OPERATORPARAMS": {"IMAG": {"Operator": compressor, "Accuracy": str(float(accuracy) / 2)}}}})
t.putcol("IMAG", value=np.imag(vis))
print(t.showstructure())
print(f"before closing: {t.getdminfo("REAL")} {t.getdminfo("IMAG")}")
t.close()

# open again
t = table(filename, readonly=True)
print(f"after reopening: {t.getdminfo("REAL")} {t.getdminfo("IMAG")}")
t.close()

on_disk_size = get_size(f'{filename}/table.f0.bp')
print(f'Table size raw: {size / 1024 / 1024:.2f} MB')
print(f'Table size on disk: {on_disk_size / 1024 / 1024:.2f} MB')
print(f'Compression ratio: {size / on_disk_size:.2f}\n\n')

# open with adios2 layer directly
print("Cross-checking with adios2 python layer:")
af = adios2.FileReader('ttable_adios_test/table.f0.bp')
print(f"Operation type: {af.inquire_variable('REAL').operations()[0].Type()}")
print(f"Operation parameters: {af.inquire_variable('REAL').operations()[0].Parameters()}")
print(f"Column accuracy: {af.inquire_variable('REAL').get_accuracy()}")

# Check the difference between original and compressed
print(f"Reading back from compressed column...")
af = adios2.FileReader('ttable_adios_test/table.f0.bp/')
visr = af.read('REAL',start=[0,0,0],count=[10000,120,4])
af.close()
af = adios2.FileReader('ttable_adios_test/table.f1.bp/')
visi = af.read('IMAG',start=[0,0,0],count=[10000,120,4])
cvis = np.complex64(visr + 1j*visi) # cast back to complex

# Plot the difference
plt.hist(np.real(vis).reshape(-1)-visr.reshape(-1), label=f'real [{accuracy}]')
plt.hist(np.imag(vis).reshape(-1)-visi.reshape(-1), label=f'imag [{float(accuracy)/2.}]')
plt.yscale("log")
plt.title(f"original-{compressor}")
plt.xlabel(f"original-{compressor}")
plt.ylabel("Log(Number) of occurance")
plt.legend()
plt.show()