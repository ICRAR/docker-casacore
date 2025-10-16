import functools
import operator
import os
import sys
import time

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

t=table('Reduced_1197634368.ms',readonly=False)
print('Read DATA')
vis=t.getcol('DATA')
#
s=vis.shape
#
cell_shape = vis.shape[1:]
size = functools.reduce(operator.mul, cell_shape, s[0] * 8)
compressor = "mgard_complex"
#compressor = "zfp"
#compressor = "sz"
accuracy = "0.1" # 1%
mode='ABS'
filename = "1197634368_casa.tab"
print(f"Will write {size / 1024 / 1024:.2f} MB of data into {filename}\n\n")

tabdesc = maketabdesc((
#        makearrcoldesc('REAL', '',
#            valuetype='float', shape=cell_shape,
#            datamanagergroup='group0', datamanagertype='Adios2StMan' ),
#        makearrcoldesc('IMAG', '',  valuetype='float', shape=cell_shape,
#                       datamanagergroup='group0', datamanagertype='Adios2StMan' ),
        makearrcoldesc('COPY', '',  valuetype='complex', shape=cell_shape,
            datamanagergroup='group0', datamanagertype='Adios2StMan' )
    ))
dminfo = makedminfo(
        tabdesc,
        {
            'group0': {
                'OPERATORPARAMS': {
#                    'IMAG': {
#                        'Operator': compressor,
#                        'mode': mode,
#                        'Accuracy': str(accuracy)},
#                    'REAL': {
#                        'Operator': compressor,
#                        'mode': mode,
#                        'Accuracy': str(accuracy)},
                    'COPY': {
                        'Operator': compressor,
                        'mode': mode,
                        'Accuracy': str(accuracy)}
               }
        } } )


#t2=table('1197634368_xml.tab',maketabdesc((macdr,macdi)),dminfo={'SPEC': {'XMLFILE':"adios_config.yaml"}})
t2=table(filename,tabdesc,dminfo=dminfo)
t2.addrows(s[0])
#t.addcols(maketabdesc((macdr,macdi))),dminfo={'SPEC': {'XMLFILE':"adios_config.yaml"}})
print('Fill COPY with DATA')
#t2.putcol('REAL',vis.real)
#t2.putcol('IMAG',vis.imag)
t2.putcol('COPY',vis)
try:
    t2.getdminfo('COPY')
except:
    print('Failed to read DMINFO for REAL')
t2.close()

print('Read back and write')
if 'COPY' in t.colnames():
  print('Remove old COPY')
  #t.removecols(['COPY'])
  t.removecols(['COPY']) #'REAL','IMAG','COPY'])
sq=[]
for n in t.colnames(): sq.append(t.getdminfo(n)['SEQNR'])
sq=np.max(np.array(sq))
#t.addcols(maketabdesc((macd)))#,'TiledShapeStMan') # This reverts to standard - should fix
#t.addcols(makearrcoldesc("REAL", 0., shape=d.shape[1:]),
#dminfo={"TYPE": "Adios2StMan", "NAME":"asm1", "SEQNR":sq+1, "SPEC": {"XMLFILE":"adios_config.yaml"}})
#t.putcol('REAL',d.real)
print('Add the data to the exisiting MS')
#t.addcols(maketabdesc((macdr,macdi,macd)),
#  dminfo={"TYPE": "Adios2StMan", "NAME":"asm1", "SEQNR":sq+1, "SPEC": {"XMLFILE":"adios_config.yaml"}})
t.addcols(tabdesc,dminfo=dminfo)
#t.putcol('REAL',vis.real);
#t.putcol('IMAG',vis.imag)
#for n in range(51):
#    t.putcol('COPY',vis[
t.putcol('COPY',vis)
a_dmi=t.getdminfo('COPY')
print('Close')
t.close()

t2=table(filename)
t=table('Reduced_1197634368.ms')#,readonly=False)
print(t2.getcol('COPY')[100,10,0],vis[100,10,0],t2.getcol('COPY')[100,10,0]-vis[100,10,0])
print(t.getcol('COPY')[1100,5,0],vis[1100,5,0],t.getcol('COPY')[5100,10,0]-vis[5100,10,0])
t.close();t2.close()
print('Query ADIOS table')
os.system('/usr/local/bin/bpls --list_operators %s/table.f0.bp'%(filename))
on_disk_size = get_size(f'{filename}/table.f0.bp')
print(f'Table size raw: {size / 1024 / 1024:.2f} MB')
print(f'Table size on disk: {on_disk_size / 1024 / 1024:.2f} MB')
print(f'Compression ratio: {size / on_disk_size:.2f}\n\n')
# open with adios2 layer directly
print("Cross-checking with adios2 python ltayer:")
af = adios2.FileReader(filename+'/table.f0.bp')
print(f"Operation type: {af.inquire_variable('COPY').operations()[0].Type()}")
print(f"Operation parameters: {af.inquire_variable('COPY').operations()[0].Parameters()}")
print(f"Column accuracy: {af.inquire_variable('COPY').get_accuracy()}")

