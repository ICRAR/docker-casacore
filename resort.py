import numpy as np ; import os,glob,shutil,json,sys
from casacore import tables
#import matplotlib.pylab as pl ;

# A quick and dirty way to reorder a MS table,
# assuming that ALL time steps and baselines are present in the row.
# That is the nRow=nTime*nBase

# Command line arguements feed in vis/outputvis and various other options
# if the file is small one can read the whole array and reorgaise it
# other wise it is read in stages (i.e. by baseline)

vis=None
outputvis=None
Auto=True
Make_Copy=False
Verbose=False
Test=False
TimeOrder=False
WOrder=False
Small_File=False

## Keep Version history upto date here
Version={}
Version['0.9']='Inital script to read entire file, order and write out (Small_File mode)'
Version['1.0']='Large files can be reordered (Small_File default to False)'
Version['1.1']='Fixed it so data is read in uvd order in large file mode'
Version['2.0']='Added TimeOrder to reverse TB to BT'
Version['2.1']='Added sort by W distance (for WSClean)'

Usage='Usage: %s [-vis=]FileIn.ms [-outputvis=]FileOut.ms [-Verbose|-Test|-Small_File|-Worder|-Make_copy|-OverWrite|-TimeOrder|-Help]'%(sys.argv[0])
if (len(sys.argv)==1):
  print(Usage)
  exit()

sysargvIn=sys.argv
remove_from_sysargv=[]
for n in sys.argv[1:]:
 if n[0]=='-':
  #n=n.replace('--','-') ## Did this to allow --What_Ever - but then it is not removed from the sysarg list
  if n.replace('-','').lower()=='help':
    print(Usage)
    for k in Version.keys(): 
      print(k+':'+Version[k])
    exit()
  if n.replace('-','').lower()=='verbose':
    print('Setting Verbose=True')
    Verbose=True
    remove_from_sysargv.append(n)
  elif (n.replace('-','').lower()=='make_copy')|(n.replace('-','').lower()=='overwrite'):
    print('Setting Make_Copy=True')
    Make_Copy=True
    remove_from_sysargv.append(n)
  elif n.replace('-','').lower()=='test':
    print('Setting Test=True')
    Test=True
    remove_from_sysargv.append(n)
  elif n.replace('-','').lower()=='small_file':
    print('Setting Small_File=True')
    Small_File=True
    remove_from_sysargv.append(n)
  elif n.replace('-','').lower()=='worder':
    print('Ordering by W-term (not uvd)')
    WOrder=True
    remove_from_sysargv.append(n)
  elif n.replace('-','').lower()=='timeorder':
    print('Setting TimeOrder=True')
    TimeOrder=True
    remove_from_sysargv.append(n)
  elif n.split('=')[0].replace('-','').lower()=='vis':
    print('Setting vis='+n.split('=')[1])
    vis=n.split('=')[1]
    remove_from_sysargv.append(n)
  elif n.split('=')[0].replace('-','').lower()=='outputvis':
    print('Setting outputvis='+n.split('=')[1])
    outputvis=n.split('=')[1]
  else: print('Unknown command line parameter: '+n)
for n in remove_from_sysargv: sys.argv.remove(n)

if vis==None:  
  vis=sys.argv[1]
  print('Setting vis='+vis)
if (len(sys.argv)>2) & (outputvis==None):
  outputvis=sys.argv[2]
  print('Setting outputvis='+outputvis)

if outputvis==None: outputvis=vis.replace('.ms','.resort.ms')

if (Make_Copy): os.system('rm -r '+outputvis)
if (os.path.isdir(outputvis)==False):
  print('Making new %s'%(outputvis))
  if Verbose:
    os.system('cp -vrp %s %s'%(vis,outputvis))
  else:
    print('Copying ',vis,' to ',outputvis)
    os.system('cp -rp %s %s'%(vis,outputvis))
else:
    #vis='1058759488.copy.ms'
    #outputvis=vis
    print('Using existing %s'%(outputvis))
if Test:
    t1=tables.table(vis)
    t2=tables.table(outputvis)
else:
    t1=tables.table(vis)
    t2=tables.table(outputvis,readonly=False)
a1=t1.getcol('ANTENNA1')
a2=t1.getcol('ANTENNA2')
uv=t1.getcol('UVW').T
t=t1.getcol('TIME')
nTime=len(np.unique(t)) # 222 in original, 4 in averaged
fg=t1.getcol('FLAG').T
uvd=np.abs(uv[0]+1j*uv[1])
if WOrder: uvd=uv[2]
nAnt=len(np.unique([a1,a2])) # 128
if (len(np.where(a1==a2)[0])>0): Auto=True
if (Auto):
    nBase=np.int32(nAnt*(nAnt+1)/2) # 8256
else:
    nBase=np.int32(nAnt*(nAnt-1)/2)
nTime=np.int32(uv[0].shape[0]/nBase)
## Not all time values are indentical post averaging

if TimeOrder: # Easiest if these are switched here
  n=nTime
  nTime=nBase
  nBase=n
  
uv=uv.reshape((3,nTime,nBase))
fg=fg.reshape((4,-1,nTime,nBase))
a1=a1.reshape((nTime,nBase))
a2=a2.reshape((nTime,nBase))
t =t.reshape((nTime,nBase))
uvd=uvd.reshape((nTime,nBase))
nChan=fg[1].shape[0]

if TimeOrder:
  #Iuvd=np.argsort(t[0]) # Sort _only_ on first baseline step
  Iuvd=np.arange(nTime)
else:
  Iuvd=np.argsort(uvd[0]) # Sort _only_ on first time step
Ifinal=np.arange((nTime*nBase)).reshape((nTime,nBase))
if TimeOrder:
    Ifinal=Ifinal[Iuvd].T.reshape((-1))
else:
    Ifinal=Ifinal[:,Iuvd].T.reshape((-1))
#pl.clf();pl.plot(uvd.reshape((-1))[Ifinal]);pl.ylabel('Ordered uv Distance')
#pl.savefig('tmp.png')

print('Time Steps %d Number Antennas %d Number Baselines %d'%(nTime,nAnt,nBase))
print(nTime*nBase,fg.shape,uv.shape,t.shape,Ifinal.shape)

#tb.putcol('ANTENNA1',a1[Ifinal])
#tb.putcol('ANTENNA2',a2[Ifinal])
#tb.putcol('UVW',uv.T[Ifinal].T)
#tb.putcol('TIME',t[Ifinal])
#tb.putcol('FLAG',fg.T[Ifinal].T)

# UVW (3, 1832832)
# FLAG (npol,nchan,ntime*nbase)
# WEIGHT (4, 1832832)
# SIGMA (4, 1832832)
# ANTENNA1 (1832832,)
# ANTENNA2 (1832832,)
# ARRAY_ID (1832832,)
# DATA_DESC_ID (1832832,)
# EXPOSURE (1832832,)
# FEED1 (1832832,)
# FEED2 (1832832,)
# FIELD_ID (1832832,)
# FLAG_ROW (1832832,)
# INTERVAL (1832832,)
# OBSERVATION_ID (1832832,)
# PROCESSOR_ID (1832832,)
# SCAN_NUMBER (1832832,)
# STATE_ID (1832832,)
# TIME (1832832,)
# TIME_CENTROID (1832832,)
# DATA (npol,nchan,ntime*nbase)
# WEIGHT_SPECTRUM (npol,nchan,ntime*nbase)

for col in t1.colnames():
    #['WEIGHT','SIGMA','ARRAY_ID','DATA_DESC_ID','EXPOSURE','FEED1','FEED2','FIELD_ID','FLAG_ROW','INTERVAL','OBSERVATION_ID','PROCESSOR_ID','SCAN_NUMBER','STATE_ID','TIME_CENTROID','ANTENNA1','ANTENNA2','TIME','UVW','TIME','DATA','WEIGHT_SPECTRUM']:
    print('Reading, reordering and resaving ',col)
    if (col=='FLAG_CATEGORY')|(col=='SIGMA_SPECTRUM'):
        print('%s not understood'%(col))
    #Alternativly s=tb.getcol(n).shape;if len(s)>1:
    elif ((col=='WEIGHT')|(col=='SIGMA')|(col=='UVW')|(col=='FLAG')|
        (col=='DATA')|(col=='WEIGHT_SPECTRUM')|
        (col=='CORRECTED_DATA')|(col=='MODEL_DATA')): # multiD 
        # Cols with (N,nBase*nTime) -- CASA only
       if Test:
         print(col,'Shape',t1.getcol(col)[Ifinal].shape)
       else:
         if Small_File==True:
            t2.putcol(col,t1.getcol(col)[Ifinal])
         else:
          for n in range(nBase):
            d=t1.getcol(col,rowincr=nBase,startrow=Iuvd[n])
            if (n==0):
              print('Shape of first read',d.shape)
            elif Verbose:
              print(n,d.shape)
            t2.putcol(col,value=d,nrow=nTime,startrow=n*nTime,rowincr=1)
    else:  # Not MultiD
       if Test:
         print(col,t1.getcol(col)[Ifinal].shape)
       else:
         if Small_File==True:
           t2.putcol(col,t1.getcol(col)[Ifinal])
         else:
          for n in range(nBase):
            d=t1.getcol(col,rowincr=nBase,startrow=Iuvd[n])
            if (n==0):
              print('Shape of first read',d.shape)
            elif Verbose:
              print(n,d.shape)
            t2.putcol(col,value=d,nrow=nTime,startrow=n*nTime,rowincr=1)
    print('Done ',col)
t1.close()
t2.close()
if Test==False:
  import time
  t2=tables.table(outputvis+'/HISTORY',readonly=False)
  n=t2.nrows() #n=-1 # Stick remarks at the end. Could loose important info ..
  t2.addrows(nrows=1)
  #d=t2.getcol('CLI_COMMAND')#,nrow=(t2.nrows()-1))
  #n=d['shape'];n[0]+=1;d['shape']=n
  #for n in range(d['shape'][0]):
  #  if d['array'][n]=='': break
  #d['array'][n]=' '.join(sysargvIn)
  #t2.putcol('CLI_COMMAND',d)
  d=t2.getcol('TIME')
  mjd=time.time()/3600/24 +40588-0.5  # Convert 01/01/1970 to MJD
  d[n]=mjd
  t2.putcol('TIME',d)
  d=t2.getcol('MESSAGE')
  if TimeOrder:
    d[n]='Resorted from Baseline Time to Time Baseline'
  else:
    d[n]='Resorted from Time Baseline to Baseline Time, with shortest first'
  t2.putcol('MESSAGE',d)
  d=t2.getcol('ORIGIN')
  k=Version.keys()
  #k.sort()
  d[n]=sysargvIn[0]#+':'+k[-1]
  t2.putcol('ORIGIN',d)
  t2.close()
