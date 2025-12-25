import numpy as np
import sys
import matplotlib.pylab as pl
from casacore.tables import ( table )
from casacore.images import ( image )

print('Opening MS ',sys.argv[-1])
tb=table(sys.argv[-1]+'/ANTENNA')
pos=tb.getcol('POSITION')
tb.close()

tb=table(sys.argv[-1])
d_old=tb.getcol('DATA')
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
            cls.append([I[m[0]],0,tmp_n1,tmp_n2])
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

d_rms=np.std(d_old)
d_diff=np.std(d_ref-d_new)
d_max=np.max(np.abs(d_ref-d_new))
#im1.close()
#im2.close()
print('Errors ',d_rms,d_diff,d_max,d_diff/d_rms,d_max/d_rms)

return(d_diff/d_rms) # return the fractional \Delta RMS over RMS 

