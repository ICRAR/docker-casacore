from casacore.tables import (makescacoldesc, makearrcoldesc, table,
                          maketabdesc)
import numpy as np

# produce some data
vis = np.random.rand(10000, 120, 4)

# create a scalar column in a new table
c1 = makescacoldesc("coli", 0)
t = table("ttable_adios_test", maketabdesc((c1)), ack=False)

# add an ADIOS2 column to that table using specs from a XML file
t.addcols(makearrcoldesc("vis1", 0., shape=vis.shape[1:]),
          dminfo={"TYPE": "Adios2StMan", "NAME":"asm1", "SPEC": {"XMLFILE":"adios.xml"}})
        #   dminfo={"TYPE": "Adios2StMan", "NAME":"asm1"})
t.addcols(makearrcoldesc("vis2", 0., shape=vis.shape[1:]),
          dminfo={"TYPE": "IncrementalStMan", "NAME": "ism1"})

# add as many rows as there are rows in the data
t.addrows(vis.shape[0])

# write the column data and close the table
t.putcol('vis1',value=vis)
t.putcol('vis2',value=vis)
t.close()

# open again and make a copy of the data
t = table("ttable_adios_test",readonly=False)
vis = t.getcol("vis1")
coldmi = t.getdminfo("vis1")
coldmi["NAME"] = "vis3"
t.addcols(maketabdesc(makearrcoldesc("vis3",0.)), coldmi)
t.putcol("vis2", value=vis)
t.putcol("vis3", value=vis)
t.close()

