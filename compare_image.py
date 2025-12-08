import numpy as np
import sys
from casacore.tables import (
    table
    )
from casacore.images import (
    image
    )


print('Opening Reference ',sys.argv[-2])
im1=image(sys.argv[-2])
d_ref=im1.getdata()
print('Opening New ',sys.argv[-1])
im2=image(sys.argv[-1])
d_new=im2.getdata()
d_rms=np.std(d_ref)
d_diff=np.std(d_ref-d_new)
d_max=np.max(np.abs(d_ref-d_new))
#im1.close()
#im2.close()
print('Errors ',d_rms,d_diff,d_max,d_diff/d_rms,d_max/d_rms)
