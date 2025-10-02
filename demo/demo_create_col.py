from casacore.tables import table, makearrcoldesc, maketabdesc
import argparse as ap
import numpy as np
import logging,adios2

t=table('Reduced_1197634368.ms',readonly=False)
print('Read DATA')
d=t.getcol('DATA')
t.close()
#
s=d.shape
macdr=makearrcoldesc('REAL',1.,2,[s[1], s[2]],'Adios2StMan') # Adios is float but not complex
macdi=makearrcoldesc('IMAG',1.,2,[s[1], s[2]],'Adios2StMan')
#
t2=table('1197634368_xml.tab',maketabdesc((macdr,macdi)),dminfo={'SPEC': {'XMLFILE':"adios_config.yaml"}})
t2.addrows(s[0])
#t.addcols(maketabdesc((macdr,macdi))),dminfo={'SPEC': {'XMLFILE':"adios_config.yaml"}})
print('Fill COPY with DATA')
#t2.putcol('REAL',d.real)
#t2.putcol('IMAG',d.imag)
# adios path
#fr=adios2.Stream('Reduced_1197634368.ms','w')
fr=adios2.Stream('1197634368_xml.tab','w')
print('Fill REAL with Adios2.Stream')
fr.write('REAL',d.real.astype(np.float32),shape=d.shape,start=(0,0,0),
         count=(s[0],s[1],s[2]),operations=[('mgard',{'accuracy':'10.0','mode':'ABS','s':'0','lossless':"Huffman_Zstd"})])
print('Fill IMAG with Adios2.Stream')
fr.write('IMAG',d.imag.astype(np.float32),shape=d.shape,start=(0,0,0),
         count=(s[0],s[1],s[2]),operations=[('mgard',{'accuracy':'10.0','mode':'ABS','s':'0','lossless':"Huffman_Zstd"})])
print('Close')
fr.close()
t2.close()
