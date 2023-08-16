import numpy as np
from decimal import Decimal, ROUND_HALF_UP

def ext_round(dat):
    if(np.size(dat)==1):
        return float(Decimal(dat).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
        
    ret = dat
    for i, x in np.ndenumerate(dat):
        ret[i] = float(Decimal(x).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

    return ret


if __name__=='__main__':
    dat = np.array([1.5,2.5,3.5,4.5])
    print(np.round(dat))

    print(ext_round(dat))