import adios2
import logging
from casacore.tables import table, tablecolumn
import argparse as ap
import numpy as np

def read_adios_write_numpy(inputfile, accuracy, column_name, cyl=True):
    if new_api:
        adios = adios2.Adios()
        io = adios.declare_io("ReadMgard")
        with adios2.Stream(io, inputfile, "r") as fr:
            nsteps = fr.num_steps()
            logger.debug(f"nsteps: {nsteps}")
            for _ in fr.steps():
                cstep = fr.current_step()
                logger.debug(f"current_step: {cstep}")
                if nsteps > 1:
                    if cstep == 0:
                        if cyl:
                            shape = fr.inquire_variable("amplitude").shape()
                            amplitude = np.empty((shape[0], nsteps,shape[1], shape[2]))
                            phase = np.empty((shape[0], nsteps, shape[1], shape[2]))
                        else:
                            shape = fr.inquire_variable("real").shape()
                            data = np.empty((shape[0], nsteps, shape[1], shape[2]), dtype=np.complex64)
                    if cyl:
                        amplitude[:, cstep, ...] = fr.read("amplitude")
                        phase[:, cstep, ...] = fr.read("phase")
                    else:
                        data[:, cstep, ...].real = fr.read("real")
                        data[:, cstep, ...].imag = fr.read("imag")
                else:
                    if cyl:
                        amplitude = fr.read("amplitude")
                        phase = fr.read("phase")
                        shape = amplitude.shape
                    else:
                        data = fr.read("real").astype(np.complex64)
                        data.imag = fr.read("imag")
                        shape = data.shape
                
            if cyl:
                np.save(f"amplitude_{column_name}_{accuracy[0]}.{cstep:%04d}.npy", np.reshape(amplitude, (shape[0], nsteps*shape[1], shape[2])))
                np.save(f"phase_{column_name}_{accuracy[1]}.{cstep:%04d}.npy", np.reshape(phase, (shape[0], nsteps*shape[1], shape[2])))
            else:
                np.save(f"data_{column_name}_{accuracy[0]}_imag_{accuracy[1]}.{cstep:%04d}.npy", np.reshape(data, (shape[0], nsteps*shape[1], shape[2])))
    else:
        with adios2.open(inputfile, 'r') as f:
            nsteps = f.steps()
            if nsteps > 1:
                vars = f.available_variables()
                if cyl:
                    shape = vars["amplitude"]["Shape"]
                    amplitude = np.empty((shape[0],nsteps,shape[1], shape[2]))
                    phase = np.empty((shape[0],nsteps,shape[1], shape[2]))
                else:
                    shape = vars["real"]["Shape"]
                    data = np.empty((shape[0],nsteps,shape[1], shape[2]), dtype=np.complex64)
                for step in f:
                    cstep = step.current_step()
                    if cyl:
                        amplitude[:, cstep, ...] = step.read("amplitude")
                        phase[: ,cstep, ...] = step.read("phase")
                    else:
                        data[:, cstep, ...].real = step.read("real")
                        data[:, cstep, ...].imag = step.read("imag")
            else:
                for step in f:
                    if cyl:
                        amplitude = step.read("amplitude")
                        phase = step.read("phase")
                    else:
                        data = step.read("real").astype(np.complex64)
                        data.imag = step.read("imag")
                        
            if cyl:
                np.save(f"amplitude_{column_name}_{accuracy[0]}.{cstep:%04d}.npy", np.reshape(amplitude, (shape[0], nsteps*shape[1], shape[2])))
                np.save(f"phase_{column_name}_{accuracy[1]}.{cstep:%04d}.npy", np.reshape(phase, (shape[0], nsteps*shape[1], shape[2])))
            else:
                np.save(f"data_{column_name}_{accuracy[0]}_imag_{accuracy[1]}.{cstep:%04d}.npy", np.reshape(data, (shape[0], nsteps*shape[1], shape[2])))
                    

def getargs() -> ap.Namespace:
    parser = ap.ArgumentParser()
    parser.add_argument("input_ms", help="Input measurement set to read from")
    parser.add_argument("column_name", help="The column to read and compress (default: DATA)", default="DATA")
    parser.add_argument("output_bp", help="The name of the output file in BP5 format (default: output.bp)", default="output.bp")
    parser.add_argument('-o','--operator', help="The operator with which to save the data (default: 'mgard')", default='mgard')
    parser.add_argument("-s", '--stepsize', help="The size of the steps (along row axis) to break the data into (default is the whole column)", default=None, type=int)
    parser.add_argument("-a", "--accuracy", help="The error bound with which to compress the data (default: 1e-4)", default=1e-4, type=float)
    parser.add_argument("-m", "--mode", help="The mode (ABS, BDC, BDC-ABS or REL) with which to apply the accuracy (default: REL)", default='REL')
    parser.add_argument("-u", "--uv_division", help="In BDC modes, increment the accuracy at this boundry",default=0,type=float)
    parser.add_argument("-nu", "--number_division", help="In BDC modes, increment in this number of sections",default=4,type=int)
    parser.add_argument('-n', '--numpy', help="Use this flag to output the data as a numpy array for each variable", action='store_true')
    parser.add_argument('-l', '--level', help="Logging Level (INFO or DEBUG)",default='INFO',type=str)


    args = parser.parse_args()
    return args

def get_column(filename: str, col: str) -> tablecolumn: 
    tab = table(filename)
    col = tab.col(col)
    return col

def read_write_data(col: tablecolumn, fcol: tablecolumn, output_name: str, stepsize: int, mode: str, accuracy: float, operator: str, uv_indx: list):
    nrows = col.nrows()
    nsteps = nrows//stepsize
    do_last_step = False
    if nsteps*stepsize < nrows:
        logger.error(f"""Steps are not balanced, {stepsize=}, {nrows=}, {nsteps=}, nsteps*stepsize={nsteps*stepsize}. 
                        Aborting now.""")
    
    # adios = adios2.Adios()
    # io = adios.declare_io("WriteMgard")
    with adios2.Stream(output_name, "w") as fr:
        step = 0
        for _ in fr.steps(nsteps):
            logger.info(f"Reading step {step}/{nsteps}")
            start = step*stepsize
            coldata = col.getcol(startrow=start, nrow=stepsize)
            fdata_bool = ~np.isfinite(coldata)  # Find and zero all NaN or Inf values
            coldata[fdata_bool] = 0.
            fdata_bool = fcol.getcol(startrow=start, nrow=stepsize) # Find and zero all FLAGed values
            coldata[fdata_bool] = 0.
            if mode == 'REL':
                accuracy0 = str(float(accuracy)*(np.nanmax(coldata.real)-np.nanmin(coldata.real)))
                accuracy1 = str(float(accuracy)*(np.nanmax(coldata.imag)-np.nanmin(coldata.imag)))
                logger.info(f"""absolute error bounds: {accuracy0}, {accuracy1}""")
                mmode = 'ABS'
            elif mode == 'BDC':
                step_acc=np.where(step>=np.array(uv_indx))[0][-1]+1
                logger.info(f"""BDC bound step {step_acc} - requested accuracy limit reduced by this factor""")
                accuracy0 = str(float(accuracy/step_acc)*(np.nanmax(coldata.real)-np.nanmin(coldata.real)))
                accuracy1 = str(float(accuracy/step_acc)*(np.nanmax(coldata.imag)-np.nanmin(coldata.imag)))
                logger.info(f"""absolute error bounds: {accuracy0}, {accuracy1}""")
                mmode = 'ABS'
            elif mode == 'BDC-ABS':
                step_acc=np.where(step>=np.array(uv_indx))[0][-1]+1
                logger.info(f"""BDC bound step {step_acc} - requested accuracy reduced by this factor""")
                accuracy0 = str(float(accuracy/step_acc))
                accuracy1 = str(float(accuracy/step_acc))
                mmode = 'ABS'
            else:
                mmode = mode
                accuracy0 = str(float(accuracy))
                accuracy1 = str(float(accuracy))
            adios_args = ({"accuracy":accuracy0, "mode":mmode, "s":"0", "lossless":"Huffman_Zstd"},
                            {"accuracy":accuracy1, "mode":mmode, "s":"0", "lossless":"Huffman_Zstd"})
            logger.debug(f"""{nrows=}
{nsteps=}
{stepsize=}
{step=}
{coldata.shape=}
{adios_args=}
{start=}""")
            logger.debug(f"""data: min_real: {np.min(coldata.real)},
max_real: {np.max(coldata.real)},
min_imag: {np.min(coldata.imag)},
max_imag: {np.max(coldata.imag)},
""")
#flagged_real%: {100.*np.sum(~np.isfinite(1./coldata.real.reshape((-1))))/np.sum(np.isfinite(coldata.real.reshape((-1))))},
#flagged_im%: {100.*np.sum(~np.isfinite(1./coldata.imag.reshape((-1))))/np.sum(np.isfinite(coldata.imag.reshape((-1))))}
            
            fr.write("real", 
                     coldata.real.astype(np.float32), 
                     shape=coldata.shape, 
                     start=(0*start, 0, 0), 
                     count=(stepsize, coldata.shape[1], coldata.shape[2]), 
                     operations=[(operator, adios_args[0])])
            fr.write("imag", 
                     coldata.imag.astype(np.float32), 
                     shape=coldata.shape, 
                     start=(0*start, 0, 0), 
                     count=(stepsize, coldata.shape[1], coldata.shape[2]), 
                     operations=[(operator, adios_args[1])])
            if args.numpy:
                print('Numpy Not Working')
                ##from compress_ms import read_adios_write_numpy
                #accuracy_input = [vars(args)["accuracy"]]
                #if len(accuracy_input)==1: accuracy_input.append(accuracy_input[0])
                #logger.info(f"Writing to numpy file")
                #read_adios_write_numpy(args.output_bp, accuracy_input, args.column_name, False)

            step+=1

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    adios_version = adios2.__version__.split('.')[:2]
    new_api = int(adios_version[0]) == 2 and int(adios_version[1]) > 9
    args = getargs()
    if args.level=='DEBUG': logging.basicConfig(level=logging.DEBUG)
    if args.level=='INFO': logging.basicConfig(level=logging.INFO)
    print(args)
    uv_indx=np.zeros((args.number_division),dtype='int')
    if args.stepsize:
      if (args.mode[:3]=='BDC'):
        col = get_column(args.input_ms, 'UVW')
        uvdata= col.getcol(startrow=0, nrow=-1, rowincr=args.stepsize)
        uvd=np.abs(uvdata.T[0]+1j*uvdata.T[1])
        if (args.uv_division>0):
            uv_trans=args.uv_division
        else:
            uv_trans=np.max(uvd)/(args.number_division+1)
        uv_indx=np.zeros((args.number_division),dtype='int')
        for n in range(args.number_division):
            uv_indx[n]=np.where(uvd>=uv_trans*n)[0][0]
        #print(uv_trans,uv_indx)
    col = get_column(args.input_ms, args.column_name)
    fcol = get_column(args.input_ms, 'FLAG')
    if not args.stepsize:
        stepsize = col.nrows()
    else:
        stepsize = args.stepsize
    read_write_data(col, fcol, args.output_bp, stepsize, args.mode, args.accuracy, args.operator, uv_indx.tolist())
