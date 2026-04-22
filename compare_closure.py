import numpy as np
import sys
import matplotlib.pylab as pl
from casacore.tables import ( table )
from casacore.images import ( image )

DataCol=sys.argv[-2]
tb=table(sys.argv[-1]+'/ANTENNA')
pos=tb.getcol('POSITION')
tb.close()

tb=table(sys.argv[-1])
if DataCol not in tb.colnames():
    DataCol='DATA'
print('Opening MS %s on %s'%(sys.argv[-1],DataCol))
d_old=tb.getcol(DataCol)
d_new=tb.getcol('COPY_DATA')
a1=tb.getcol('ANTENNA1')
a2=tb.getcol('ANTENNA2')
time=tb.getcol('TIME')
uvw=tb.getcol('UVW')

pl.clf();pl.plot(pos.T[0],pos.T[1],'o');pl.savefig('tmp.png')
cls=[]
data_cls_val=[]
vis_cls_val=[]
diff_cls_val=[]
for t in np.unique(time):
    I=np.where(time==t)[0]
    n0=0
    for n1 in range(1,127):
      for n2 in range(n1+1,128):
        #print(n1,n2)
        m=np.where(((a1[I]==n1)|(a1[I]==n2))&((a2[I]==n1)|(a2[I]==n2))&(a1[I]!=a2[I]))
        if len(m[0])==1:
            tmp_n1=n1
            tmp_n2=n2
            if np.sum(a1[I[m[0]]]==n2):
                tmp_n2=-n1
                tmp_n1=-n2
            cls.append([I[m[0]][0],0,tmp_n1,tmp_n2])
            m0=np.where((a1[I]==n0)&(a2[I]==n1))[0]
            m1=np.where((a1[I]==n0)&(a2[I]==n2))[0]
            tmp_n0=np.nanmean(d_old[I[m0]],axis=(0,1))
            tmp_n1=np.nanmean(d_old[I[m[0]]],axis=(0,1))
            tmp_n2=np.nanmean(d_old[I[m1]],axis=(0,1))
            data_cls_val.append([tmp_n0,tmp_n1,np.conj(tmp_n2)])
            tmp_n0=np.nanmean(d_new[I[m0]],axis=(0,1))
            tmp_n1=np.nanmean(d_new[I[m[0]]],axis=(0,1))
            tmp_n2=np.nanmean(d_new[I[m1]],axis=(0,1))
            vis_cls_val.append([tmp_n0,tmp_n1,np.conj(tmp_n2)])
            diff_cls_val.append(np.sum(np.angle(vis_cls_val[-1]),axis=0)-np.sum(np.angle(data_cls_val[-1]),axis=0))
    if (t==np.unique(time)[0])&(np.mod(n2,10)==0)&(np.mod(n1,10)==0):
        Ip=cls[-1][1:]
        Ip.append(Ip[0])
        pl.plot(pos[Ip,0],pos[Ip,1],'k-')
pl.savefig('tmp.png')
vis_cls_val=np.array(vis_cls_val)
data_cls_val=np.array(data_cls_val)
cls=np.array(cls)

d_max=np.nanmax(data_cls_val-vis_cls_val)
d_diff=np.nanstd(data_cls_val-vis_cls_val)
c_max=np.nanmax(np.angle(np.exp(-1j*(np.angle(data_cls_val)-np.angle(vis_cls_val))))*57.3)
c_diff=np.nanstd(np.angle(np.exp(-1j*(np.angle(data_cls_val)-np.angle(vis_cls_val))))*57.3)
print('Closure Error (deg): ',d_diff,d_max,c_diff,c_max)
#d_max=np.nanmax(data_cls_val-vis_cls_val,axis=0)
#d_diff=np.nanstd(data_cls_val-vis_cls_val,axis=0)
#c_max=np.nanmax(np.angle(np.exp(-1j*(np.angle(data_cls_val)-np.angle(vis_cls_val))))*57.3,axis=0)
#c_diff=np.nanstd(np.angle(np.exp(-1j*(np.angle(data_cls_val)-np.angle(vis_cls_val))))*57.3,axis=0)
#print('Closure Error (deg): ')
#n0=d_max.shape
#for n1 in range(n0[0]):
#    #for n2 in range(n0[1]):
#       print(d_diff[n1])
#for n1 in range(n0[0]):
#    #for n2 in range(n0[1]):
#       print(d_max[n1])
#for n1 in range(n0[0]):
#    #for n2 in range(n0[1]):
#       print(c_diff[n1])
#for n1 in range(n0[0]):
#    #for n2 in range(n0[1]):
#       print(c_max[n1])

#d_rms=np.std(d_old)
#d_diff=np.std(d_old-d_new)
#d_max=np.max(np.abs(d_old-d_new))
##im1.close()
##im2.close()
#print('Errors ',d_rms,d_diff,d_max,d_diff/d_rms,d_max/d_rms)

##return(d_diff/d_rms) # return the fractional \Delta RMS over RMS 

