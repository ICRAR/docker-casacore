import adios2
from casacore.tables import table, makearrcoldesc, maketabdesc
import argparse as ap
import numpy as np
import logging

def getargs() -> ap.Namespace:
    parser = ap.ArgumentParser()
    parser.add_argument("input_bp", help="The input bp file, usually compressed")
    parser.add_argument("output_table", help="The casa table to output the data into")
    parser.add_argument('-l', '--level', help="Logging Level (INFO or DEBUG)",default='INFO',type=str)
    parser.add_argument('-n', '--new', help="New table otherwise reuse",default=False,type=bool)

    return parser.parse_args()

def read_write_data() -> None:
    logger.info(f"Input .bp file: {args.input_bp}")
    logger.info(f"Output casa table: {args.output_table}")
    with adios2.Stream(args.input_bp, 'r') as fr:
        
        step = 0
        for _ in fr.steps():
            logger.info(f"Reading step {step}")
            nsteps = fr.num_steps()
            if step == 0:
                vars = fr.available_variables()
                shape = [int(x) for x in vars['real']['Shape'].split(',')]
                element_shape = shape[1:]
                coldesc = makearrcoldesc('DATA', 1+1j, len(element_shape), element_shape, 'TiledShapeStMan')
                tabdesc = maketabdesc([coldesc])
                if args.new:
                    tab = table(args.output_table, tabdesc, readonly=False)
                else:
                    tab = table(args.output_table, readonly=False)
            if args.new:
                tab.addrows(shape[0])
            data = fr.read('real').astype(np.complex64)
            data.imag = fr.read('imag')
            logger.debug(f"""data: min_real: {np.min(data.real)},
                                    max_real: {np.max(data.real)},
                                    min_imag: {np.min(data.imag)},
                                    max_imag: {np.max(data.imag)},
    """)
            
            if args.new:
                tab.putcol('DATA', data, step*shape[0])
            else:
                tab.putcol('DATA', data, startrow=step*shape[0], nrow=shape[0])
            step += 1
        tab.close()

            

if __name__ =='__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    args = getargs()
    if args.level=='DEBUG': logging.basicConfig(level=logging.DEBUG)
    if args.level=='INFO': logging.basicConfig(level=logging.INFO)    
    logger.info("Reading and writing data")
    read_write_data()
