import sys
import numpy as np
from casacore.tables import (
  makearrcoldesc,
  makescacoldesc,
  table,
  maketabdesc,
  makedminfo)


print(sys.argv)
DATACOL='DATA'

tb=table(sys.argv[-2],readonly=False)
print('Working on ',sys.argv[-2])

SHAP=np.array(tb.getcol(DATACOL,nrow=1).shape)
SHAP[0]=tb.nrows()
cell_shape=SHAP[1:]

if 'ORIG_DATA' not in tb.colnames():
    dminfo=tb.getdminfo(DATACOL)
    dminfo['NAME']='ORIG_DATA'
    Dtabdesc=tb.getdesc()[DATACOL]
    Ttabdesc = maketabdesc(
        (makearrcoldesc('ORIG_DATA', '',
            valuetype=Dtabdesc['valueType'], shape=cell_shape,
            datamanagergroup='group1', datamanagertype='TiledShapeStMan' ),
         ))

    tb.addcols(Ttabdesc,dminfo=dminfo)
    for n in range(193):
        print('Making %d/193 Copy to ORIG_DATA'%(n))
        tb.putcol('ORIG_DATA',tb.getcol(DATACOL,nrow=9990,startrow=n*9990),nrow=9990,startrow=n*9990)

tb2=table(sys.argv[-1])
print('Replacing from ',sys.argv[-1])
for n in range(193):
    print('Making %d/193 replacing DATA'%(n))
    tb.putcol(DATACOL,tb2.getcol('COPY',nrow=9990,startrow=n*9990),nrow=9990,startrow=n*9990)

tb.close()
tb2.close()
