import numpy as np
import scipy


def interp3_spline(V,k):
    """
    matlab function interp3(V,k,'spline'), spline interpolation
    :param V: sample data, (n1,n2,n3) tensor
    :param k: refinement factor, this results in 2^k-1 interpolated points between sample values.
    :return: interpolated values, (2**k*(n1-1)+1,2**k*(n1-1)+1,2**k*(n1-1)+1) tensor
    """
    n1 = np.size(V,0)
    n2 = np.size(V,1)
    n3 = np.size(V,2)

    # generate sample grid points
    x = np.linspace(0,n1-1,n1)
    y = np.linspace(0,n2-1,n2)
    z = np.linspace(0,n3-1,n3)

    # generate query coordinates at each dimension
    xi = np.linspace(0,n1-1,2**k*(n1-1)+1)
    yi = np.linspace(0,n2-1,2**k*(n2-1)+1)
    zi = np.linspace(0,n3-1,2**k*(n3-1)+1)

    # call interpolator
    fn = scipy.interpolate.RegularGridInterpolator((x,y,z), V, 'cubic')

    # generate xyz coordinates respectively
    xx = np.repeat(xi[:,None,None],len(yi),axis=1)
    xxx = np.repeat(xx,len(zi),axis=2)
    yy = np.repeat(yi[None,:,None],len(xi),axis=0)
    yyy = np.repeat(yy,len(zi),axis=2)
    zz = np.repeat(zi[None,None,:],len(xi),axis=0)
    zzz = np.repeat(zz,len(yi),axis=1)

    # concatenate coordinates
    coors0 = np.concatenate((xxx[:,:,:,None],yyy[:,:,:,None],zzz[:,:,:,None]),axis=-1)

    # calculate interpolated values
    val = np.zeros((2**k*(n1-1)+1,2**k*(n2-1)+1,2**k*(n3-1)+1))
    for ii in range(2**k*(n1-1)+1):
        for jj in range(2**k*(n2-1)+1):
            for kk in range(2**k*(n3-1)+1):
                val[ii,jj,kk] = fn(coors0[ii,jj,kk,:])

    return val

if __name__ == '__main__':
    import time
    from interp_cubic_convolution import *

    # generate random sample data with size(13,13,20)
    ref1 = np.random.randn(4,4,4)

    # print runtime for interp3_cubic_convolution
    startT = time.perf_counter()
    refimhd = interp3_cubic_convolution(ref1,2)
    endT = time.perf_counter()
    print('interp3_cubic_convolution used {:.3f} s.'.format((endT-startT)))

    # print runtime for interp3_spline
    startT = time.perf_counter()
    refimhd = interp3_spline(ref1,2)
    endT = time.perf_counter()
    print('interp3_spline used {:.3f} s.'.format((endT-startT)))

    # print runtime for scipy.ndimage.zoom
    k = 2
    startT = time.perf_counter()
    n1 = np.size(ref1,0)
    n2 = np.size(ref1,1)
    n3 = np.size(ref1,2)
    refimhd1 = scipy.ndimage.zoom(ref1, (((2**k*(n1-1)+1)/n1,(2**k*(n2-1)+1)/n2,(2**k*(n3-1)+1)/n3)), mode='grid-constant', grid_mode=True)
    # refimhd2 = scipy.ndimage.zoom(ref1, 2.0, mode='grid-constant', grid_mode=True)
    # refimhd3 = scipy.ndimage.zoom(ref1, 3.0, mode='grid-constant', grid_mode=True)
    # refimhd4 = scipy.ndimage.zoom(ref1, 4.0, mode='grid-constant', grid_mode=True)
    endT = time.perf_counter()
    print('scipy.ndimage.zoom used {:.3f} s.'.format((endT-startT)))

    print(np.max(refimhd-refimhd1))