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
datacol='DATA'
filename='1197639168.ms'
filename='Reduced_1197634368.ms'
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
compressor = "zfp"
#compressor = "sz"
accuracy = "%f"%(rms*1e-3) # 1%
mode='ABS'
#
if 'COPY' in t.colnames():
  print('Remove old COPY')
  t.removecols('COPY')
if 'COPY_STD' in t.colnames():
  print('Remove old standard COPY_STD')
  t.removecols('COPY_STD')
if 'COPY_1D' in t.colnames():
  print('Remove old 1D COPY')
  t.removecols('COPY_1D')

tabdesc_std = maketabdesc(
        (makearrcoldesc('COPY_STD', '',
            valuetype='complex', shape=cell_shape,
            datamanagergroup='std2', datamanagertype='StandardStMan' ),
         ))
tabdesc = maketabdesc(
        (makearrcoldesc('COPY', '',
            valuetype='complex', shape=cell_shape,
            datamanagergroup='groupA', datamanagertype='Adios2StMan' ),
         ))
tabdesc_1d = maketabdesc(
        (makearrcoldesc('COPY_1D', '',
            valuetype='complex', shape=[s[1]*s[2]],
            datamanagergroup='groupB', datamanagertype='Adios2StMan' ),
         ))
dminfo = makedminfo(
        tabdesc,
        {
            'groupA': {  # Now with no compression
                 # 'OPERATORPARAMS': {
                 #   'COPY': {
                 #       'Operator': compressor,
                 #       'mode': mode,
                 #       'Accuracy': str(accuracy)} }
               } } )
dminfo_std = makedminfo(
        tabdesc_std,
        {
            'std2': {  # Now with no compression
                 # 'OPERATORPARAMS': {
                 #   'COPY': {
                 #       'Operator': compressor,
                 #       'mode': mode,
                 #       'Accuracy': str(accuracy)} }
               } } )
dminfo_1d = makedminfo(
        tabdesc_1d,
        {
            'groupB': {  # Now with no compression
                 # 'OPERATORPARAMS': {
                 #   'COPY': {
                 #       'Operator': compressor,
                 #       'mode': mode,
                 #       'Accuracy': str(accuracy)} }
               } } )


t.addcols(tabdesc,dminfo=dminfo)
t.addcols(tabdesc_std,dminfo=dminfo_std)
t.addcols(tabdesc_1d,dminfo=dminfo_1d)
a_dmi=t.getdminfo('COPY')
print('DM:',a_dmi)
print('Fill COPY with DATA')
output_name='%s/table.f%d.bp'%(filename,a_dmi['SEQNR'])
#fr=adios2.Stream(output_name,'a')
#fr.steps(Ntime)
adios_args={"accuracy":str(float(accuracy)), "mode":mode, "s":"0", "lossless":"Huffman_Zstd"}

for n in range(Ntime):
    vis=t.getcol(datacol,nrow=Nbase,startrow=n*Nbase)
    s=vis.shape
    print('Read %d/%d of shape %d (rms %f)\t'%(n,Ntime,vis.shape[0],np.nanstd(vis[I,:,0])))
    if compressor == "mgard_complex":
        vis=vis.reshape(-1)
        Inan=np.where(np.isnan(vis)==True)[0]
        vis[Inan]*=0
        vis=vis.reshape(s)
        print('Write %d/%d with %d NaNs (rms %f)\t'%(n,Ntime,len(Inan),np.std(vis)))
    t.putcol('COPY',vis,nrow=Nbase,startrow=n*Nbase)
    t.putcol('COPY_STD',vis,nrow=Nbase,startrow=n*Nbase)
    t.putcol('COPY_1D',vis.reshape((-1,s[1]*s[2])),nrow=Nbase,startrow=n*Nbase)
    #fr.write("complex",vis,
    #                 shape=vis.shape, 
    #                 start=(0, 0, 0), 
    #                 count=(Nbase, s[1], s[2]), 
    #                 operations=[(compressor, adios_args)])
t.close()

t=table(filename,readonly=False)
for n in range(Ntime):
    data=t.getcol(datacol,nrow=Nbase,startrow=n*Nbase)
    vis=t.getcol('COPY',nrow=Nbase,startrow=n*Nbase)
    print('COPY %d/%d (rms error %f)\t'%(n,Ntime,np.nanstd((vis-data)[I,:,0])))
    vis=t.getcol('COPY_STD',nrow=Nbase,startrow=n*Nbase)
    print('COPY_STD %d/%d (rms error %f)\t'%(n,Ntime,np.nanstd((vis-data)[I,:,0])))
    vis=t.getcol('COPY_1D',nrow=Nbase,startrow=n*Nbase)
    print('COPY_1D %d/%d (rms error %f)\t'%(n,Ntime,np.nanstd((vis-data.reshape((-1,s[1]*s[2]))))))
t.removecols('COPY')
t.addcols(tabdesc,dminfo=dminfo)
t.putcol('COPY',t.getcol(datacol))
t.close()
t=table(filename)
print('Errors after single write:',np.nanstd(t.getcol(datacol)-t.getcol('COPY')))


print('Close')
t.close()
