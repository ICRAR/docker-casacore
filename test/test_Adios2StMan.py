#!/usr/bin/env python
from casacore.tables import table
from glob import glob
from numpy import array, complex64
import shutil

d = array(
    [[ 0.+0.j,  1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j],
     [ 5.+0.j,  6.+0.j,  7.+0.j,  8.+0.j,  9.+0.j],
     [10.+0.j, 11.+0.j, 12.+0.j, 13.+0.j, 14.+0.j],
     [15.+0.j, 16.+0.j, 17.+0.j, 18.+0.j, 19.+0.j],
     [20.+0.j, 21.+0.j, 22.+0.j, 23.+0.j, 24.+0.j],
     [25.+0.j, 26.+0.j, 27.+0.j, 28.+0.j, 29.+0.j]], dtype=complex64)
t = table('/scratch/default.table')
t = table('/scratch/duplicated.table')
data = t.getcol(t.colnames()[0])[0]
t.close()

assert(d.all() == data.all())
print("Cleaning up...")
ms = glob("*.table")
shutil.rmtree("/scratch/default.table")
shutil.rmtree("/scratch/duplicated.table")