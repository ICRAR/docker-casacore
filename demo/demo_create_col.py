from casacore.tables import table, makearrcoldesc, maketabdesc
import argparse as ap
import numpy as np
import logging,adios2,os

t=table('Reduced_1197634368.ms',readonly=False)
print('Read DATA')
d=t.getcol('DATA')
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
         count=(s[0],s[1],s[2]),operations=[('mgard',{'accuracy':'1.0','mode':'ABS','s':'0','lossless':"Huffman_Zstd"})])
print('Fill IMAG with Adios2.Stream')
fr.write('IMAG',d.imag.astype(np.float32),shape=d.shape,start=(0,0,0),
         count=(s[0],s[1],s[2]),operations=[('mgard',{'accuracy':'1.0','mode':'ABS','s':'0','lossless':"Huffman_Zstd"})])
print('Close')
fr.close()
print('Read back and write')
fr=adios2.Stream('1197634368_xml.tab','r')
if 'COPY' in t.colnames():
  print('Remove old COPY')
  t.removecols('COPY')
macd=makearrcoldesc('COPY',1+1j,2,[s[1],s[2]],'TiledShapeStMan')
t.addcols(maketabdesc((macd)))#,'TiledShapeStMan') # This reverts to standard - should fix
fr=adios2.Stream('1197634368_xml.tab','r')
for _ in fr.steps():
    var=fr.available_variables()
    data=fr.read('REAL').astype(np.complex64)
    data.imag=fr.read('IMAG')
    t.putcol('COPY',data) # If we have steps these will need adding here
print('Close')
fr.close()
t.getdminfo('COPY')
t.close()
try:
    t2.getdminfo('REAL')
except:
    print('Failed to read DMINFO for REAL')
t2.close()

print('Query ADIOS table')
os.system('/usr/local/bin/bpls --list_operators 1197634368_xml.tab')




os.system('rm -r 1197634368_pycasa.tab')
t2=table('1197634368_pycasa.tab',maketabdesc((macdr,macdi)),dminfo={'SPEC': {'XMLFILE':"adios_config.yaml"}})
t2.addrows(s[0])
os.system('ls 1197634368_pycasa.tab')
print(t2.getdminfo('REAL'))
print('Fill COPY with DATA')
try:
    t2.putcol('REAL',d.real)
    t2.putcol('IMAG',d.imag)
except:
    print('Failed to read back for REAL/IMAG')
t2.close()

print('Query ADIOS table')
os.system('/usr/local/bin/bpls --list_operators 1197634368_pycasa.tab')


os.system('rm -r 1197634368_pycasa0.tab')
t2=table('1197634368_pycasa0.tab',maketabdesc((macdr,macdi)))#,dminfo={'NAME':'Adios2StMan','SPEC': {'XMLFILE':"adios_config.yaml"}})
t2.addrows(s[0])
os.system('ls 1197634368_pycasa.tab/')
print(t2.getdminfo('REAL'))

print('Fill COPY with DATA')
try:
    t2.putcol('REAL',0*d.real)
    t2.putcol('IMAG',0*d.imag)
except:
    print('Failed to read back for REAL/IMAG')
t2.close()

print('Query ADIOS table')
os.system('du -sh  1197634368_*.tab')
