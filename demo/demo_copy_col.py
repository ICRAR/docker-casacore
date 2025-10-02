from casacore.tables import table, makearrcoldesc, maketabdesc
import argparse as ap
import numpy as np
import logging

t=table('Reduced_1197634368.ms',readonly=True)
print('Read DATA')
d=t.getcol('DATA')
#
s=d.shape
macdr=makearrcoldesc('REAL',1.,2,[s[1], s[2]],'Adios2StMan') # Adios is float but not complex
macdi=makearrcoldesc('IMAG',1.,2,[s[1], s[2]],'Adios2StMan')
t2=table('1197634368_xml.tab',maketabdesc((macdr,macdi)),dminfo={'SPEC': {'XMLFILE':"adios_m.xml"}})
t2.addrows(t.nrows())
print('Fill COPY with DATA')
t2.putcol('REAL',d.real)
t2.putcol('IMAG',d.imag)
t2.close()
#
t2=table('1197634368_yaml.tab',maketabdesc((macdr,macdi)),dminfo={'SPEC': {'XMLFILE':"adios_config.yaml"}})
t2.addrows(t.nrows())
print('Fill COPY with DATA')
t2.putcol('REAL',d.real)
t2.putcol('IMAG',d.imag)
t2.close()
#
macd=makearrcoldesc('COPY',1+1j,2,[s[1],s[2]],'StandardStMan')
t2=table('1197634368_std.tab',maketabdesc(macd))
t2.addrows(t.nrows())
print('Fill COPY with DATA')
t2.putcol('COPY',d)
t2.close()
