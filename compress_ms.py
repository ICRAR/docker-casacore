import adios2
from casacore.tables import table
import argparse as ap
import numpy as np
import shutil
import os
import logging
# from mpi4py import MPI

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

def get_column_data(filename, column='DATA'):
    ms = table(filename)
    data = ms.getcol(column)
    return data

# def write_adios_full(data, output_bp, operator: str="mgard", accuracy=("1e-6","1e-6"), mode: str="ABS", s: str="0", cyl = True, stepwise=False):
#     adios = adios2.ADIOS()
#     io = adios.DeclareIO("Output")
#     engine = io.Open(output_bp, adios2.Mode.Write)
#     nchans = data.shape[1]
#     engine.BeginStep()

#     operators = (adios.DefineOperator(f"{operator}0", operator, {"accuracy":accuracy[0], "mode":mode, "s":s, "lossless":"Huffman_Zstd"}),
#                  adios.DefineOperator(f"{operator}1", operator, {"accuracy":accuracy[1], "mode":mode, "s":s, "lossless":"Huffman_Zstd"}))

#     if cyl:
#         ampdata = np.abs(data)
#         phdata = np.angle(data)
#         vars = {"amplitude":ampdata, "phase":phdata}
#     else:
#         vars = {"real":data.real, "imag":data.imag}
#     for i, (varname, var) in enumerate(vars.items()):
#         variable = io.DefineVariable(varname, var.astype(np.float32))
#         # variable.AddOperation(operators[i], {})
#         print(var)
#         for chan in range(nchans):
        
#             engine.Put(arg0=variable,arg1='test')
#     engine.PerformPuts()
#     engine.EndStep()
#     engine.Close()
    
def write_adios_high(data, output_bp, operator: str="mgard", accuracy=("1e-6","1e-6"), mode: str="ABS", s: str="0", cyl = True, stepwise=False):
    # if stepwise, extract num chans
    if stepwise:
        dshape = data.shape
        nsteps = dshape[1]
        start = [0, mystart, 0]
        count = [dshape[0], 1, dshape[2]]
        shape = [dshape[0], nchans, dshape[2]]
    else:
        dshape = data.shape
        nsteps = 1
        start = [0, mystart, 0]
        count = [dshape[0], dshape[1], dshape[2]]
        shape = [dshape[0], nchans, dshape[2]]
    logger.debug(f"""
                 dshape: {dshape}
                 nsteps: {nsteps}
                 start: {start}
                 count: {count}
                 shape: {shape}
                 """)
    fdata_bool = ~np.isfinite(data)
    # rand_data = np.random.rand(fdata_bool.sum())*100-50
    data[fdata_bool] = 0.
    # logger.info(f"""random data range: {np.min(rand_data)}, {np.max(rand_data)}""")
    
    if new_api:
        adios = adios2.Adios()
        io = adios.declare_io("WriteMgard")
        with adios2.Stream(io, output_bp, "w") as fr:
            i = 0
            for _ in fr.steps(nsteps):
                logger.info(f"Writing step {i}")
                if cyl:
                    if stepwise:
                        ampdata = np.abs(data[:, i, :])
                        phdata = np.angle(data[:, i, :])
                    else:
                        ampdata = np.abs(data)
                        phdata = np.angle(data)
                    if mode == 'REL':
                        accuracy0 = str(float(accuracy[0])*(np.nanmax(ampdata)-np.nanmin(ampdata)))
                        accuracy1 = str(float(accuracy[1])*(np.nanmax(phdata)-np.nanmin(phdata)))
                        logger.info(f"""absolute error bounds: {accuracy0}, {accuracy1}""")
                        mmode = 'ABS'
                    else:
                        mmode = mode
                        accuracy0 = accuracy[0]
                        accuracy1 = accuracy[1]
                    adios_args = ({"accuracy":accuracy0, "mode":mmode, "s":s, "lossless":"Huffman_Zstd"},
                                  {"accuracy":accuracy1, "mode":mmode, "s":s, "lossless":"Huffman_Zstd"})
                    logger.info(f"""adios_args: {adios_args},
                                data: min_amp: {np.nanmin(ampdata)},
                                    max_amp: {np.nanmax(ampdata)},
                                    min_phase: {np.nanmin(phdata)},
                                    max_phase: {np.nanmax(phdata)}""")
                    fr.write("amplitude", ampdata.astype(np.float32), shape=shape, start=list(start),count=list(count), operations=[(operator, adios_args[0])])
                    fr.write("phase", phdata.astype(np.float32), shape=shape, start=list(start),count=list(count), operations=[(operator, adios_args[1])])
                else:
                    if stepwise:
                        sdata = data[:, i, :]
                    else:
                        sdata = data
                    if mode == 'REL':
                        accuracy0 = str(float(accuracy[0])*(np.nanmax(sdata.real)-np.nanmin(sdata.real)))
                        accuracy1 = str(float(accuracy[1])*(np.nanmax(sdata.imag)-np.nanmin(sdata.imag)))
                        logger.info(f"""absolute error bounds: {accuracy0}, {accuracy1}""")
                        mmode = 'ABS'
                    else:
                        mmode = mode
                        accuracy0 = accuracy[0]
                        accuracy1 = accuracy[1]
                    adios_args = ({"accuracy":accuracy0, "mode":mmode, "s":s, "lossless":"Huffman_Zstd"},
                                  {"accuracy":accuracy1, "mode":mmode, "s":s, "lossless":"Huffman_Zstd"})
                    logger.info(f"""adios_args: {adios_args},
                                data: min_real: {np.nanmin(sdata.real)},
                                    max_real: {np.nanmax(sdata.real)},
                                    min_imag: {np.nanmin(sdata.imag)},
                                    max_imag: {np.nanmax(sdata.imag)}""")
                    fr.write("real", sdata.real.astype(np.float32), shape=shape, start=list(start),count=list(count), operations=[(operator, adios_args[0])])
                    fr.write("imag", sdata.imag.astype(np.float32), shape=shape, start=list(start),count=list(count), operations=[(operator, adios_args[1])])
                i+=1
    else:
        with adios2.open(output_bp, 'w') as f:
            for i in range(nsteps):
                logger.info(f"Writing step {i}")
                if cyl:
                    if stepwise:
                        ampdata = np.abs(data[:, i, :])
                        phdata = np.angle(data[:, i, :])
                    else:
                        ampdata = np.abs(data)
                        phdata = np.angle(data)
                    if mode == 'REL':
                        accuracy0 = str(float(accuracy[0])*(np.nanmax(ampdata)-np.nanmin(ampdata)))
                        accuracy1 = str(float(accuracy[1])*(np.nanmax(phdata)-np.nanmin(phdata)))
                        logger.info(f"""absolute error bounds: {accuracy0}, {accuracy1}""")
                        mmode = 'ABS'
                    else:
                        mmode = mode
                        accuracy0 = accuracy[0]
                        accuracy1 = accuracy[1]
                    adios_args = ({"accuracy":accuracy0, "mode":mmode, "s":s, "lossless":"Huffman_Zstd"},
                                  {"accuracy":accuracy1, "mode":mmode, "s":s, "lossless":"Huffman_Zstd"})
                    logger.info(f"""adios_args: {adios_args},
                                data: min_amp: {np.nanmin(ampdata)},
                                    max_amp: {np.nanmax(ampdata)},
                                    min_phase: {np.nanmin(phdata)},
                                    max_phase: {np.nanmax(phdata)}""")
                    # print("amplitude", ampdata.astype(np.float32), data.shape, start,count, [(operator, {"accuracy":accuracy, "mode":mode, "s":s, "lossless":"Huffman_Zstd"})])
                    f.write("amplitude", ampdata.astype(np.float32), shape=shape, start=start,count=count, operations=[(operator, adios_args[0])])
                    # print("phase", phdata.astype(np.float32), data.shape, start,count, [(operator, {"accuracy":accuracy, "mode":mode, "s":s, "lossless":"Huffman_Zstd"})])
                    f.write("phase", phdata.astype(np.float32), shape=shape, start=start,count=count, operations=[(operator, adios_args[1])])
                else:
                    if stepwise:
                        sdata = data[:, i, :]
                    else:
                        sdata = data
                    if mode == 'REL':
                        accuracy0 = str(float(accuracy[0])*(np.nanmax(sdata.real)-np.nanmin(sdata.real)))
                        accuracy1 = str(float(accuracy[1])*(np.nanmax(sdata.imag)-np.nanmin(sdata.imag)))
                        logger.info(f"""absolute error bounds: {accuracy0}, {accuracy1}""")
                        mmode = 'ABS'
                    else:
                        mmode = mode
                        accuracy0 = accuracy[0]
                        accuracy1 = accuracy[1]
                    adios_args = ({"accuracy":accuracy0, "mode":mmode, "s":s, "lossless":"Huffman_Zstd"},
                                  {"accuracy":accuracy1, "mode":mmode, "s":s, "lossless":"Huffman_Zstd"})
                    logger.info(f"""adios_args: {adios_args},
                                data: min_real: {np.nanmin(sdata.real)},
                                    max_real: {np.nanmax(sdata.real)},
                                    min_imag: {np.nanmin(sdata.imag)},
                                    max_imag: {np.nanmax(sdata.imag)}""")
                    # print("real", data.real.astype(np.float32), data.shape, start,count, [(operator, {"accuracy":accuracy, "mode":mode, "s":s, "lossless":"Huffman_Zstd"})])
                    f.write("real", sdata.real.astype(np.float32), shape=shape, start=start,count=count, operations=[(operator, adios_args[0])])
                    # print("imag", data.imag.astype(np.float32), data.shape, start,count, [(operator, {"accuracy":accuracy, "mode":mode, "s":s, "lossless":"Huffman_Zstd"})])
                    f.write("imag", sdata.imag.astype(np.float32), shape=shape, start=start,count=count, operations=[(operator, adios_args[1])])

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
                np.save(f"amplitude_{column_name}_{accuracy[0]}.npy", np.reshape(amplitude, (shape[0], nsteps*shape[1], shape[2])))
                np.save(f"phase_{column_name}_{accuracy[1]}.npy", np.reshape(phase, (shape[0], nsteps*shape[1], shape[2])))
            else:
                np.save(f"data_{column_name}_{accuracy[0]}_imag_{accuracy[1]}.npy", np.reshape(data, (shape[0], nsteps*shape[1], shape[2])))
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
                np.save(f"amplitude_{column_name}_{accuracy[0]}.npy", np.reshape(amplitude, (shape[0], nsteps*shape[1], shape[2])))
                np.save(f"phase_{column_name}_{accuracy[1]}.npy", np.reshape(phase, (shape[0], nsteps*shape[1], shape[2])))
            else:
                np.save(f"data_{column_name}_{accuracy[0]}_imag_{accuracy[1]}.npy", np.reshape(data, (shape[0], nsteps*shape[1], shape[2])))
                    
def read_adios_write_ms(inputfile, input_ms, column_name, accuracy,cyl=True):
    inputname = os.path.basename(input_ms)
    output_name = "_".join((inputname,column_name, accuracy[0],accuracy[0]))
    shutil.copytree(input_ms, output_name)
    
    ms = table(output_name,readonly=False)
    if new_api:
        adios = adios2.Adios()
        io = adios.declare_io("ReadMgard")
        with adios2.Stream(io, inputfile, "r") as fr:
            for _ in fr.steps():
                if cyl:
                    amplitude = fr.read("amplitude")
                    phase = fr.read("phase")
                    data = amplitude*np.cos(phase).astype(np.complex64)
                    data.imag = amplitude*np.sin(phase)
                else:
                    data = fr.read("real").astype(np.complex64)
                    data.imag = fr.read("imag")
                ms.putcol(column_name, data)
                    
                    
    else:
        with adios2.open(inputfile, 'r') as f:
            for step in f:
                if cyl:
                    amplitude = step.read("amplitude")
                    np.save(f"amplitude_{column_name}_{accuracy[0]}.npy", amplitude)
                    phase = step.read("phase")
                    np.save(f"phase_{column_name}_{accuracy[1]}.npy", phase)
                else:
                    data = step.read("real").astype(np.complex64)
                    data.imag = step.read("imag")
                    np.save(f"data_{column_name}_{accuracy[0]}_imag_{accuracy[1]}.npy", data)

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("input_ms", help="The path to the input Measurement Set, this works for any casacore table")
    parser.add_argument("column_name", help="The name of the column that contains the data to compress")
    parser.add_argument("output_bp", help="The output file path")
    parser.add_argument('-f','--full',help="Write the data using the full adios python API. Default is to use the High-level API.", action='store_true')
    parser.add_argument('-o','--operator', help="The operator with which to save the data (default: 'mgard')", default='mgard')
    parser.add_argument('-a','--accuracy', help="The accuracy parameter for the operator, provide 2 values if two accuracies are required (default: '1e-6')", default='1e-6', nargs='+')
    parser.add_argument('-m','--mode', help="The mode parameter for the operator, 'ABS' or 'REL' (default: 'ABS')",default='ABS')
    parser.add_argument('-s','--smoothness', help="The smoothness parameter for the operator (default: 0)",default='0')
    parser.add_argument('-c', '--cylindrical', help="Use amplitude and phase instead of real and imaginary", action='store_true')
    parser.add_argument('-n', '--numpy', help="Use this flag to output the data as a numpy array for each variable", action='store_true')
    parser.add_argument('-r', '--replace', help="Copy the input MeasurementSet and replace the column with the compressed data", action='store_true')
    parser.add_argument('-t', '--stepwise', help="Write the data stepped along the channel axis, i.e. compress each channel individually. (Currently not working)", action='store_true')
    
    args = parser.parse_args()
    if args.mode not in ('ABS','REL'):
        print(f"--mode must be either 'ABS' or 'REL', {args.mode} entered.")
        raise AssertionError
    if len(args.accuracy) == 1:
        args.accuracy.append(args.accuracy[0])
    if args.full:
        raise NotImplementedError("Writing using the full API is currently broken, please use the high-level api.")
    if args.stepwise:
        raise NotImplementedError("Stepwise compression is under construction and currently not working")
    return args

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # check if new or old api
    adios_version = adios2.__version__.split('.')[:2]
    new_api = int(adios_version[0]) == 2 and int(adios_version[1]) > 9
    args = parse_args()
    logger.info(f"Reading Column data: {args.column_name}")
    data = get_column_data(args.input_ms, args.column_name)
    nchans = data.shape[1]
    chuncksize = nchans
    mystart = 0
    myend = chuncksize
    start = mystart
    end = myend
    mydata = data[:, mystart:myend, :]
    # if args.full:
    #     write_adios_full(data, args.output_bp, args.operator, args.accuracy, args.mode, args.smoothness, args.cylindrical, args.stepwise)
    logger.info(f"""
                Writing data using adios
                output: {args.output_bp}
                operator: {args.operator}
                accuracy: {args.accuracy}
                mode: {args.mode}
                smoothness: {args.smoothness}
                """)
    accuracy_input = tuple(vars(args)["accuracy"])
    write_adios_high(mydata, args.output_bp, args.operator, args.accuracy, args.mode, args.smoothness, args.cylindrical, args.stepwise)
    if args.numpy:
        logger.info(f"Writing to numpy file")
        read_adios_write_numpy(args.output_bp, accuracy_input, args.column_name, args.cylindrical)
    if args.replace:
        logger.info(f"Writing to MeasurementSet")
        read_adios_write_ms(args.output_bp, args.input_ms, args.column_name,args.accuracy,args.cylindrical)
    
    