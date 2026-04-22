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

datacol='CORRECTED_DATA'
filename='1197639168.ms'
t=table(filename,readonly=False)
print('Read DATA')
vis=t.getcol(datacol,nrow=1)
#
s=vis.shape
#
cell_shape = vis.shape[1:]
Nrows=t.nrows()
Ntime=len(np.unique(t.getcol('TIME')))
Nbase=int(Nrows/Ntime)
size = functools.reduce(operator.mul, cell_shape, t.nrows() * 8)
#
print(f"Will write {size / 1024 / 1024:.2f} MB of data into {filename}\n\n")
#t.close()
#t=table('1197639168.ms',readonly=False)
# I am assuming all baselines are there in the first record and don't change ... 
a1=t.getcol('ANTENNA1',nrow=Nbase,startrow=0)
a2=t.getcol('ANTENNA2',nrow=Nbase,startrow=0)
I=np.where(a1!=a2)[0]
rms=np.nanstd(t.getcol(datacol,nrow=Nbase,startrow=0)[I,:,1])
print('Data RMS: %.3e'%(rms))
#
compressor = "mgard_complex"
#compressor = "zfp"
#compressor = "sz"
accuracy = "%f"%(rms*1e-3) # 1%
mode='ABS'
#
if 'COPY' in t.colnames():
  print('Remove old standard COPY')
  t.removecols('COPY')

tabdesc = maketabdesc(
        (makearrcoldesc('COPY', '',
            valuetype='complex', shape=cell_shape,
            datamanagergroup='group0', datamanagertype='Adios2StMan' ),
       # makearrcoldesc('IMAG', '',  valuetype='float', shape=cell_shape,
       #     datamanagergroup='group0', datamanagertype='Adios2StMan'
         ))
dminfo = makedminfo(
        tabdesc,
        {
            'group0': {
                'OPERATORPARAMS': {
               #     'IMAG': {
               #         'Operator': compressor,
               #         'mode': mode,
               #         'Accuracy': str(accuracy)},
                    'COPY': {
                        'Operator': compressor,
                        'mode': mode,
                        'Accuracy': str(accuracy)}
               } } } )


#t2=table('1197634368_xml.tab',maketabdesc((macdr,macdi)),dminfo={'SPEC': {'XMLFILE':"adios_config.yaml"}})
t.addcols(tabdesc,dminfo=dminfo)
#t.addcols(maketabdesc(makearrcoldesc('COPY',1.+1j,2,cell_shape)))
#t.addcols(tabdesc,dminfo=dminfo)
#a_dmi=t.getdminfo('REAL')
#
#t2=table(filename,tabdesc,dminfo=dminfo)
#t2.addrows(Nrows)
#t.addcols(maketabdesc((macdr,macdi))),dminfo={'SPEC': {'XMLFILE':"adios_config.yaml"}})
#try:
#    print(t2.getdminfo('REAL'))
#except:
#    print('Failed to read DMINFO for REAL')

print('Read back and write')
#t.addcols(maketabdesc((macd)))#,'TiledShapeStMan') # This reverts to standard - should fix
#t.addcols(makearrcoldesc("REAL", 0., shape=d.shape[1:]),
#dminfo={"TYPE": "Adios2StMan", "NAME":"asm1", "SEQNR":sq+1, "SPEC": {"XMLFILE":"adios_config.yaml"}})
#t.putcol('REAL',d.real)
print('Add the data to the exisiting MS')
#t.addcols(maketabdesc((macdr,macdi,macd)),
#  dminfo={"TYPE": "Adios2StMan", "NAME":"asm1", "SEQNR":sq+1, "SPEC": {"XMLFILE":"adios_config.yaml"}})

print('Fill COPY with DATA')

for n in range(Ntime):
    vis=t.getcol(datacol,nrow=Nbase,startrow=n*Nbase)
    s=vis.shape
    print('Read %d/%d of %d (rms %f)\t'%(n,Ntime,vis.shape[0],np.nanstd(vis[I,:,0])))
    vis=vis.reshape(-1)
    Inan=np.where(np.isnan(vis)==True)[0]
    vis[Inan]=0
    vis=vis.reshape(s)
    #t.putcol('REAL',vis.real,nrow=Nbase,startrow=n*Nbase);
    #t.putcol('IMAG',vis.imag,nrow=Nbase,startrow=n*Nbase)
    t.putcol('COPY',vis,nrow=Nbase,startrow=n*Nbase)
    print('Write %d/%d of %d)\t'%(n,Ntime,len(Ibase)))
    #t2.putcol('IMAG',vis.imag,nrow=Nbase,startrow=n*Nbase)
t.close()
#t.close()
#t2=table(filename)
#print('Make Standard')
#t3=table(filename_std,maketabdesc(makearrcoldesc('COPY',1.+1j,2,cell_shape,valuetype='complex')))
#t3.addcols(makearrcoldesc('COPY',1.+1j,2,cell_shape))
#t3.addrows(Nrows)

t=table(filename)
for n in range(Ntime):
    tic=time.time()
    data=t.getcol(datacol,nrow=Nbase,startrow=n*Nbase)
    tsteps = time.time()-tic
    print(f'Read {Nbase} compressed complex visibilities from {datacol} column in {tsteps:.3f}s')
    tic=time.time()
    vis=t.getcol('COPY',nrow=Nbase,startrow=n*Nbase)
    tsteps = time.time()-tic
    print(f'Read {Nbase} compressed complex visibilities from COPY column in {tsteps:.3f}s')
    print('Read %d/%d (rms %f)\t'%(n,Ntime,np.nanstd((vis-data)[I,:,0])))
for n in range(1):
    tic=time.time()
    data=t.getcol(datacol,nrow=Nbase*Ntime,startrow=0)
    print(f'Read {Nbase} compressed complex visibilities from {datacol} column in {tsteps:.3f}s')
    tic=time.time()
    vis=t.getcol('COPY',nrow=Nbase*Ntime,startrow=0)
    tsteps = time.time()-tic
    print(f'Read {Nbase} compressed complex visibilities from COPY column in {tsteps:.3f}s')
    print('Read %d/1 (rms %f)\t'%(n,np.nanstd((vis-data)[I,:,0])))
    data=t.getcol(datacol,nrow=Nbase,startrow=n*Nbase)
    vis=t.getcol('COPY',nrow=Nbase,startrow=n*Nbase)
    print('Read %d/%d (rms %f)\t'%(n,Ntime,np.std((vis-data)[I,:,0])))
print('Close')
t.close()

print('Query ADIOS table')
os.system('/usr/local/bin/bpls --list_operators %s/table.f0.bp'%(filename))
on_disk_size = get_size(f'{filename}/table.f0.bp')
print(f'Table size raw: {size / 1024 / 1024:.2f} MB')
print(f'Table size on disk: {on_disk_size / 1024 / 1024:.2f} MB')
print(f'Compression ratio: {size / on_disk_size:.2f}\n\n')
# open with adios2 layer directly
print("Cross-checking with adios2 python layer:")
af = adios2.FileReader(filename+'/table.f0.bp')
print(f"Operation type: {af.inquire_variable('COPY').operations()[0].Type()}")
print(f"Operation parameters: {af.inquire_variable('COPY').operations()[0].Parameters()}")
print(f"Column accuracy: {af.inquire_variable('COPY').get_accuracy()}")


