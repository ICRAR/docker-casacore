for f in 1*68.ms ; do
    echo 'WSClean Reference: ' $f ; 
    export ff=`echo $f|sed s/.ms//` ; 
    docker run -it -v $PWD:/scratch icrar/wsclean wsclean\ -auto-threshold\ 1\ -auto-mask\ 3\ -data-column\ CORRECTED_DATA\ -size\ 8000\ 8000\ -name\ ${ff}-DATA\ -scale\ 15arcsec\ -niter\ 20000\ ${f} ; 
    for acc in 1 5 10 50 100 500; do 
	 echo 'Compressing ' $f ' at ' $acc ; 
	 docker run -it -v $PWD:/scratch icrar/wsclean python\ create_compressed_col.py\ --filename=$f\ --compressor=zfp\ --accuracy=$acc\ --datacol=CORRECTED_DATA ;   #--datacol=CORRECTED_DATA --step=51 --delete ; 
	 echo 'WSClean Compressed: ' $f ; 
	 docker run -it -v $PWD:/scratch icrar/wsclean wsclean\ -auto-threshold\ 1\ -auto-mask\ 3\ -data-column\ COPY_DATA\ -size\ 8000\ 8000\ -name\ ${ff}-${acc}-COPY_DATA\ -scale\ 15arcsec\ -niter\ 20000\ ${f} ; 
	 docker run -it -v $PWD:/scratch icrar/wsclean python\ compare_image.py\ ${ff}-DATA-image.fits\ ${ff}-${acc}-COPY_DATA-image.fits ;
     done ;
done	 
