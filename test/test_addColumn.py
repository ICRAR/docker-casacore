from casacore.tables import (makescacoldesc, makearrcoldesc, table,
                          maketabdesc, tableexists, tableiswritable,
                          tableinfo, tablefromascii, tabledelete,
                          makecoldesc, msconcat, removeDerivedMSCal,
                          taql, tablerename, tablecopy, tablecolumn,
                          addDerivedMSCal, removeImagingColumns,
                          addImagingColumns, complete_ms_desc,
                          required_ms_desc, tabledefinehypercolumn,
                          default_ms, default_ms_subtable, makedminfo)
import numpy as np
import collections

c1 = makescacoldesc("coli", 0)
c2 = makescacoldesc("cold", 0.)
c3 = makescacoldesc("cols", "")
c4 = makescacoldesc("colb", True)
c5 = makescacoldesc("colc", 0. + 0j)
c6 = makearrcoldesc("colarr", 0.)
t = table("ttable.py_tmp.tab1", maketabdesc((c1, c2, c3, c4, c5, c6)), ack=False)

t.addcols(maketabdesc(makescacoldesc("coli2", 0)),
          dminfo={'TYPE': "Adios2StMan", 'NAME': "asm1",
          'SPEC': {'XMLFILE':"adios.xml"}})
t.addrows(3)
t.putcol('coli2',value=np.array([20,21,22]))
t.close()
t = table("ttable.py_tmp.tab1")
print(t.getcol('coli2'))