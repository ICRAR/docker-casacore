from casacore.tables import table
import os,sys
#acc=int(sys.argv[-1])
#comp='sz'
#NewName=f'COPY_DATA_{comp}_{acc}'
NewName=sys.argv[-1]
FileName=sys.argv[-2]
print(f'Renaming COPY_DATA to {NewName} in {FileName}')
tb=table(FileName,readonly=False)
tb.renamecol('COPY_DATA',NewName)
tb.close()
