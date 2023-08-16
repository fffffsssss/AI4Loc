from cv2 import CAP_PROP_XI_TEST_PATTERN_GENERATOR_SELECTOR
import numpy as np
from scipy import misc, fftpack
import time

from .interp3_spline import *
from .interp_cubic_convolution import *
from .ext_round import *
from .interp3_0 import *


def get3Dcorrshift(refim, targetim, maxmethod='interp'):
    if(np.sum(refim)==0 or np.sum(targetim)==0):
        shift = np.zeros(1,3)
        CC = 0
        return shift,CC

    sim = np.shape(targetim)

    if(np.size(sim)<3 or sim[2]==1):
        print('shift, CC = get2Dcorrshift(refim, targetim, maxmethod)')

    nanval = -1
    refimnn = refim
    refimnn[np.isnan(refimnn)]=nanval
    tarimnn = targetim
    tarimnn[np.isnan(targetim)]=nanval

    refimhd = interp3_cubic_convolution(refimnn,2)
    targetimhd = interp3_cubic_convolution(tarimnn,2)

    n = np.shape(refimhd) # not used

    reffft2 = np.fft.fft2(refimhd.transpose([2,0,1])).transpose(1,2,0)
    reffft = np.fft.fft(reffft2,axis=2)

    targetfft2 = np.fft.fft2(targetimhd.transpose([2,0,1])).transpose(1,2,0)
    targetfft = np.fft.fft(targetfft2,axis=2)

    Gfft = reffft*np.conj(targetfft)
    Gfft_ifft2 = np.fft.ifft2(Gfft.transpose([2,0,1])).transpose(1,2,0)
    Gfft_ifft2_ifft = np.fft.ifft(Gfft_ifft2,axis=2)
    Gs = np.fft.fftshift(np.real(Gfft_ifft2_ifft))

    Gs[Gs<0] = np.nan

    nisn = ~np.isnan(targetimhd)

    G = Gs/((np.nanmean(refimhd))*np.nanmean(targetimhd)*np.sum(nisn))-1
    CC = np.max(G)

    if(maxmethod =='fit'):
        maxind=[getmaxFit(np.sum(np.sum(G,2),3),3),getmaxFit(np.sum(np.sum(G,3),1),3),getmaxFit(np.sum(np.sum(G,1),2),3)];
    else:
        ind=np.argmax(G)
        if ind>1:
            [x,y,z]=mat3_ind2sub(np.shape(G),ind)
            maxind=getmaxInterp(G,np.array([x,y,z]),.05,1)
        else:
            maxind=(np.size(refim)*2+1)

    shift=np.array(maxind)/4-(np.array(np.shape(refim))-1)/2

    return shift, CC

def mat3_ind2sub(mat3_shape, ind):
    # x*mat3_shape[1]*mat3_shape[2] + y*mat3_shape[2] + z = ind
    x = (ind.astype(np.int32) // (mat3_shape[1]*mat3_shape[2]) )
    yz_ind = ind % (mat3_shape[1]*mat3_shape[2])
    y = (yz_ind.astype(np.int32) // (mat3_shape[2]))
    z = yz_ind % mat3_shape[2]
    return (x, y, z)

# def ind2sub(array_shape, ind):
#     ind[ind<0] = -1
#     ind[ind>=array_shape[0]*array_shape[1]] = -1
#     rows = (ind.astype(np.int32) / array_shape[1])
#     cols = ind % array_shape[1]
#     return (rows, cols)

# def sub2ind(array_shape, rows, cols):
#     ind = rows*array_shape[1] + cols
#     ind[ind<0] = -1
#     ind[ind>=array_shape[0]*array_shape[1]] = -1
#     return ind

def getmaxInterp(V,pos,dx,w):
    Vhd = interp3_cubic_convolution_grid(V,pos,dx,w)
    ind = np.argmax(Vhd)
    [x,y,z]=mat3_ind2sub(np.shape(Vhd),ind)
    maxind=[x*dx+pos[0]-w, y*dx+pos[1]-w ,z*dx+pos[2]-w]

    return maxind

def getmaxFit(in_dat,window=5):
    sw = ext_round((window-1)/2)
    ind = np.argmax(in_dat)
    range_l = max(0,ind-sw)
    range_r = min(ind+sw,np.size(in_dat))
    
    print('off') # warning('off') in matlab

    ps = np.squeeze(in_dat[range_l:range_r])
    [psx, S] = np.polyfit(range,ps,2)
    
    maxind = -psx(2)/2/psx(1)

    print('on') # warning('on') in matlab

