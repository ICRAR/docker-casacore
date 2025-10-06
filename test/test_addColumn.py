import functools
import operator
import os

from casacore.tables import (makescacoldesc, makearrcoldesc, table,
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
nrows = 10000
vis = np.random.rand(nrows, 120, 4)
cell_shape = vis.shape[1:]
size = functools.reduce(operator.mul, cell_shape, nrows * 8)

# various settings
compressor = "mgard"
accuracy = "0.1"
filename = "ttable_adios_test"
print(f"Will write {size / 1024 / 1024:.2f} MB of data into {filename}\n\n")

# setup table
tabdesc = maketabdesc(
        makearrcoldesc('IMAG', '',
            valuetype='double', shape=cell_shape,
            datamanagergroup='group0', datamanagertype='Adios2StMan'
        )
    )

# setup dminfo with compression
dminfo = makedminfo(
        tabdesc,
        {
            'group0': {
                'OPERATORPARAMS': {
                    'IMAG': {
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
t.putcol('IMAG',value=vis)
cmi = t.getdminfo()
print(t.showstructure())
t.close()

on_disk_size = get_size(f'{filename}/table.f0.bp')
print(f'Table size raw: {size / 1024 / 1024:.2f} MB')
print(f'Table size on disk: {on_disk_size / 1024 / 1024:.2f} MB')
print(f'Compression ratio: {size / on_disk_size:.2f}\n\n')

# open with adios2 layer directly
print("Cross-checking with adios2 python layer:")
af = adios2.FileReader('ttable_adios_test/table.f0.bp')
print(f"Operation type: {af.inquire_variable('IMAG').operations()[0].Type()}")
print(f"Operation parameters: {af.inquire_variable('IMAG').operations()[0].Parameters()}")
print(f"Column accuracy: {af.inquire_variable('IMAG').get_accuracy()}")
