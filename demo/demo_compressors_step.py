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
t=table('1197639168.ms',readonly=False)
#datacol='DATA'
#t=table('demo/Reduced_1197634368.ms',readonly=False)
print('Read DATA')
vis=t.getcol(datacol,nrow=1)
#
s=vis.shape
#
cell_shape = vis.shape[1:]
Nrows=t.nrows()
Ntime=len(np.unique(t.getcol('TIME')))
#Ntime=1
Nbase=int(Nrows/Ntime)
size = functools.reduce(operator.mul, cell_shape, t.nrows() * 8)
#
compressor = "zfp" #"mgard"
mode='REL'
filename = "1197639168/1197639168_%s.tab"%(compressor)
#
print(f"Will write {size / 1024 / 1024:.2f} MB of data into {filename}\n\n")
#t.close()
#t=table('1197639168.ms',readonly=False)
# I am assuming all baselines are there in the first record and don't change ... 
a1=t.getcol('ANTENNA1',nrow=Nbase,startrow=0)
a2=t.getcol('ANTENNA2',nrow=Nbase,startrow=0)
I=np.where(a1!=a2)[0]
rms=np.nanstd(t.getcol(datacol,nrow=Nbase,startrow=0)[I,:,1:3]) # cross pols cross correl
RMS=10.
print('Data RMS: %.3e'%(rms))
accuracy = "%f"%(rms*1e-3) # 1%
#
if 'REAL' in t.colnames():
  print('Remove old COPY')
  t.removecols(['REAL','IMAG','COPY'])
#if 'COPY' in t.colnames():
#  print('Remove old standard COPY')
#  t.removecols('COPY')

tabdesc = maketabdesc(
        (makearrcoldesc('REAL', '',
            valuetype='float', shape=cell_shape,
            datamanagergroup='group0', datamanagertype='Adios2StMan' ),
        makearrcoldesc('IMAG', '',  valuetype='float', shape=cell_shape,
            datamanagergroup='group0', datamanagertype='Adios2StMan'),
        makearrcoldesc('COPY', '',  valuetype='complex', shape=cell_shape,
            datamanagergroup='group0', datamanagertype='Adios2StMan'))
)
tabdesc = maketabdesc(
         (makearrcoldesc('REAL', '',
             valuetype='float', shape=cell_shape,
              datamanagertype='StandardStMan' ),
         makearrcoldesc('IMAG', '',  valuetype='float', shape=cell_shape,
              datamanagertype='StandardStMan'),
         makearrcoldesc('COPY', '',  valuetype='complex', shape=cell_shape,
              datamanagertype='StandardStMan'))
)
tabdesc_r = maketabdesc(
        (makearrcoldesc('REAL', '',
            valuetype='float', shape=cell_shape,
            datamanagergroup='group0', datamanagertype='Adios2StMan' )))
tabdesc_i = maketabdesc(
        (makearrcoldesc('IMAG', '',
            valuetype='float', shape=cell_shape,
            datamanagergroup='group1', datamanagertype='Adios2StMan' )))
tabdesc_c = maketabdesc(
        (makearrcoldesc('COPY', '',
            valuetype='float', shape=cell_shape,
            datamanagergroup='group2', datamanagertype='Adios2StMan' )))

dminfo = makedminfo(
        tabdesc,
        {
            'group0': {
                'OPERATORPARAMS': {
                    'REAL': {
                        'Operator': compressor,
                        'mode': mode,
                        'Accuracy': str(accuracy)},
                    'IMAG': {
                        'Operator': compressor,
                        'mode': mode,
                        'Accuracy': str(accuracy)},
                    'COPY': {
                        'Operator': compressor,
                        'mode': mode,
                        'Accuracy': str(accuracy)}
                        
               } } } )

dminfo_r = makedminfo(
        tabdesc,
        {
            'group0': {
                'OPERATORPARAMS': {
                    'REAL': {
                        'Operator': compressor,
                        'mode': mode,
                        'Accuracy': str(accuracy)} }}})
dminfo_i = makedminfo(
        tabdesc,
        {
            'group1': {
                'OPERATORPARAMS': {
                    'IMAG': {
                        'Operator': compressor,
                        'mode': mode,
                        'Accuracy': str(accuracy)} }}})
dminfo_c = makedminfo(
        tabdesc,
        {
            'group2': {
                'OPERATORPARAMS': {
                    'COPY': {
                        'Operator': compressor,
                        'mode': mode,
                        'Accuracy': str(accuracy)} }}})


#t2=table('1197634368_xml.tab',maketabdesc((macdr,macdi)),dminfo={'SPEC': {'XMLFILE':"adios_config.yaml"}})
t.addcols(tabdesc_r,dminfo=dminfo_r)
t.addcols(tabdesc_i,dminfo=dminfo_i)
t.addcols(tabdesc_c,dminfo=dminfo_c)
t2=table(filename,tabdesc,dminfo=dminfo)
t2.addrows(Nrows)
#t.addcols(maketabdesc((macdr,macdi))),dminfo={'SPEC': {'XMLFILE':"adios_config.yaml"}})
try:
    print(t.getdminfo('REAL'))
except:
    print('Failed to read DMINFO for REAL')

print('Add the data to the exisiting MS')
print('Fill REAL/IMAG with DATA')
for n in range(Ntime):
    vis=t.getcol(datacol,nrow=Nbase,startrow=n*Nbase)
    s=vis.shape
    print('Read %d/%d of %d (rms %f)\t'%(n,Ntime,vis.shape[0],np.nanstd(vis[I,:,0])))
    if True: #operator=="mgard":
        fg=t.getcol('FLAG',nrow=Nbase,startrow=n*Nbase)
        vis=vis.reshape(-1)
        fg=fg.reshape(-1)
        Inan=np.where(np.isnan(vis)==True)[0]
        vis[Inan]=0
        fg[Inan]=True
        Inan=np.where(fg==True)[0]
        vis[Inan]=0
        vis=vis.reshape(s)
        fg=fg.reshape(s)
        t.putcol('FLAG',fg,nrow=Nbase,startrow=n*Nbase);
    #vis=np.random.rand(s[0],s[1],s[2])+1j*np.random.rand(s[0],s[1],s[2])
    print('After filtering mean %f rms %f\t'%(np.mean(vis),np.std(vis)))
    vis=vis*0
    t.putcol('REAL',1.+vis.real,nrow=Nbase,startrow=n*Nbase);
    t.putcol('IMAG',1.+vis.imag,nrow=Nbase,startrow=n*Nbase)
    t2.putcol('REAL',1.+vis.real,nrow=Nbase,startrow=n*Nbase)
    t2.putcol('IMAG',1.+vis.imag,nrow=Nbase,startrow=n*Nbase)
t2.close()
#t.close()
t2=table(filename)
#print('Make Standard')
#t3=table(filename_std,maketabdesc(makearrcoldesc('COPY',1.+1j,2,cell_shape,valuetype='complex')))
#t3.addcols(makearrcoldesc('COPY',1.+1j,2,cell_shape))
#t3.addrows(Nrows)

for n in range(Ntime):
    vis=(t2.getcol('REAL',nrow=Nbase,startrow=n*Nbase)+1j*t2.getcol('IMAG',nrow=Nbase,startrow=n*Nbase)).astype('complex')
    t.putcol('COPY',vis,nrow=Nbase,startrow=n*Nbase)
    #t3.putcol('COPY',vis,nrow=Nbase,startrow=n*Nbase)                
    print('Write %d/%d (rms %f)\t'%(n,Ntime,np.nanmean(vis[I,:,0])))

print('Close')
t.close()
t2.close()

print('Query ADIOS table')
os.system('/usr/local/bin/bpls --list_operators %s/table.f0.bp'%(filename))
on_disk_size = get_size(f'{filename}/table.f0.bp')
print(f'Table size raw: {size / 1024 / 1024:.2f} MB')
print(f'Table size on disk: {on_disk_size / 1024 / 1024:.2f} MB')
print(f'Compression ratio: {size / on_disk_size:.2f}\n\n')
# open with adios2 layer directly
print("Cross-checking with adios2 python layer:")
af = adios2.FileReader(filename+'/table.f0.bp')
print(f"Operation type: {af.inquire_variable('IMAG').operations()[0].Type()}")
print(f"Operation parameters: {af.inquire_variable('IMAG').operations()[0].Parameters()}")
print(f"Column accuracy: {af.inquire_variable('IMAG').get_accuracy()}")


