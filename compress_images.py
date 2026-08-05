def compress_image(fitsimage=None,imagename=None,ACC='0.0001',COMP='sz'):
  """
  Given fitsimage and no imagename a CASA image will be made from the FITS image
  The imagename can be derived FITS name
  If a fitsimage name is given then a new CASA image is made from that

  The function will generate a new map_adios column from the map column, with compression
  """
  import numpy as np
  from casacore.tables import (table,taql,maketabdesc,makedminfo,makearrcoldesc)
  from casacore.images import image
  
  if (imagename==None)&(fitsimage!=None): 
      imagename=fitsimage.replace('fits','casa')
      print(f'Imagename will be {imagename}')
  if (fitsimage!=None):
      print(f'Creating {imagename} from {fitsimage}')
      fp=image(fitsimage)
      fp.saveas(imagename,overwrite=True) 
  elif (imagename==None):
      print(f'I will need at least one of {fitsimage} or {imagename}')
      exit()
  tb2=table(imagename,readonly=False)
  map=tb2.getcol('map')
  if len(map.shape)==5:
    SHAP=[map.shape[1],map.shape[2],map.shape[3],map.shape[4]]
  elif len(map.shape)==4:
    SHAP=[map.shape[1],map.shape[2],map.shape[3]]
  elif len(map.shape)==3:
    SHAP=[map.shape[1],map.shape[2]]
  else:
    print('Dont understand the image shape',map.shape)
    exit()
  Atabdesc = maketabdesc(
          (makearrcoldesc('map_adios', '',
              valuetype='float', shape=SHAP,
              datamanagergroup='group0', datamanagertype='Adios2StMan' ),
           ));
  Adminfo = makedminfo(Atabdesc,
          {
              'group0': {
                  'OPERATORPARAMS': {
                      'map_adios': {
                          'Operator': COMP,
                          'mode': 'ABS',
                          'Accuracy': ACC,
                          'lossless_type' : 'huffman_zstd'
                   }}}})
  if 'map_adios' in tb2.colnames(): tb2.removecols('map_adios')
  tb2.addcols(Atabdesc,dminfo=Adminfo)
  tb2.getdminfo()
  tb2.putcol('map_adios',map,nrow=1,startrow=0)
  tb2.close()
  tb2=table(imagename,readonly=True)
  mapa=tb2.getcol('map_adios')
  print('Difference for ',COMP,ACC,np.max(np.abs(mapa-map)))
  tb2.close()
