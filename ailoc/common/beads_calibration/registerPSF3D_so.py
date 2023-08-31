import numpy as np
from scipy.optimize import *
import scipy.io as scio
import copy
import os

from .ext_round import *
from .get3Dcorrshift import *
from .interp3_0 import *
from .bsarray import *


def registerPSF3D_so(imin,p,axs,filenumber):
    if('xrange' not in p):
        p['xrange']=np.arange(0,np.size(imin,0))
    if('yrange' not in p):
        p['yrange']=np.arange(0,np.size(imin,1))
    if('framerange' not in p):
        p['framerange']=np.arange(0,np.size(imin,2))

    numbeads=np.size(imin,3)

    if numbeads==1:
        imout=imin
        shiftedstackn=imin
        shift=np.array([0,0,0])
        indgood=np.array([True])
        return imout,shiftedstackn,shift,indgood

    if(np.size(p['zshiftf0'])!=0):
        zshiftf0 = p['zshiftf0']
    else:
        zshiftf0 = np.zeros([numbeads,1])

    imina = imin

    numref = 1
    avim = np.nanmean(imina[:,:,:,p['sortind'][0:numref]],3)
    ph = copy.copy(p)
    lcc = int(np.ceil((np.min([13,len(p['yrange'])])-1)/2))
    mp = int(np.ceil(((len(p['yrange']))-1)/2)+1)
    ph['yrange'] = p['yrange'][(mp-lcc-1):(mp+lcc)] # range to be check
    ph['xrange'] = p['xrange'][(mp-lcc-1):(mp+lcc)] # range to be check
    ph['framerange'] = np.arange(0,np.size(avim,2)) # range to be check

    shiftedstack,shift,cc=aligntoref(avim,imina, zshiftf0,ph)

    # calculate good ones
    shiftedstackn = normalizstack(shiftedstack, p)
    indgood = np.full((1,np.size(shiftedstackn,3)), True, dtype=bool)[0]
    indgood,res1,normamp1,co1,cc1 = getoverlap(shiftedstackn, shift, ph, indgood)
    meanim = np.nanmean(shiftedstack[:,:,:,indgood],3)
    meanim[np.isnan(meanim)] = avim[np.isnan(meanim)]

    ph['framerange'] = p['framerange']
    shiftedstack,shift2,cc=aligntoref(meanim,shiftedstack, 0*zshiftf0,ph)

    shift = shift+shift2

    shiftedstackn=normalizstack(shiftedstack, p)

    indgood = np.full((1,np.size(shiftedstackn,3)), True, dtype=bool)[0]
    indgood,res2,normamp2,co2,cc2 = getoverlap(shiftedstackn, shift, ph, indgood)
    indgood,res3,normamp3,co3,cc3 = getoverlap(shiftedstackn, shift, ph, indgood)
    indgood,res,normglobal,co,cc2 = getoverlap(shiftedstackn, shift, ph, indgood)
    shiftedstackn=shiftedstackn/normglobal

    imout = np.nanmean(shiftedstackn[:,:,:,indgood], 3)
    shiftedstackn[0,-1,:,np.logical_not(indgood)]=np.nanmax(shiftedstackn)
    shiftedstackn[0,:,1,np.logical_not(indgood)]=np.nanmax(shiftedstackn)

    return imout,shiftedstackn,shift,indgood

def getoverlap(shiftedstackn,shift,p,indgood):
    shiftedstackn11 = shiftedstackn[np.int32(p['xrange']),:,:,:]
    shiftedstackn12 = shiftedstackn11[:,np.int32(p['yrange']),:,:]
    shiftedstackn13 = shiftedstackn12[:,:,np.int32(p['framerange']),:]
    shiftedstackn14 = shiftedstackn13[:,:,:,indgood]
    refimn = np.nanmean(shiftedstackn14,3)
    cc = np.zeros((np.size(shiftedstackn,3),1))
    for k in range(np.size(shiftedstackn,3)-1,-1,-1):
        shiftedstackn21 = shiftedstackn[np.int32(p['xrange']),:,:,:]
        shiftedstackn22 = shiftedstackn21[:,np.int32(p['yrange']),:,:]
        shiftedstackn23 = shiftedstackn22[:,:,np.int32(p['framerange']),:]
        imh = shiftedstackn23[:,:,:,k]
        
        badind = np.logical_or(np.isnan(imh),np.isnan(refimn))
        cc[k] = np.sum(refimn[np.logical_not(badind)]*imh[np.logical_not(badind)])/(np.sum(refimn[np.logical_not(badind)])*np.sum(imh[np.logical_not(badind)]))*np.sum(np.logical_not(badind))
    cc = cc.flatten()

    normamp = np.nanmax(refimn)
    shiftedstackn = shiftedstackn/normamp
    refimn = refimn/normamp
    res = np.zeros((np.size(shiftedstackn,3),1))
    for k in range(np.size(shiftedstackn,3)-1,-1,-1):
        shiftedstackn31 = shiftedstackn[np.int32(p['xrange'][1:-1]),:,:,:]
        shiftedstackn32 = shiftedstackn31[:,np.int32(p['yrange'][1:-1]),:,:]
        shiftedstackn33 = shiftedstackn32[:,:,np.int32(p['framerange']),:]
        sim = shiftedstackn33[:,:,:,k]
        
        dv = (refimn[1:-1,1:-1,:]-sim)**2
        res[k] = np.sqrt(np.nanmean(dv))
    res = res.flatten()

    rescc = res/cc
    rescc[np.logical_or(np.abs(shift[:,0])>3,np.abs(shift[:,1])>3)]=np.nan
    [a,b] = robustMean([rescc[cc>0]])
    if(np.isnan(b)):
        a = np.nanmean(rescc)
        b = np.nanstd(rescc)

    co = a+2.5*b
    indgood = rescc<=co

    return indgood,res,normamp,co,cc

def robustMean(data,dim=[],k=3,fit=0):
    # different with MATLAB: ignore index output
    if(dim==[]):
        if(len(np.shape(data))==2 and np.size(data,0)==1):
            dim = 2
        else:
            dim = 1
    dim = dim-1

    if(fit==1):
        if(np.sum(np.shape(data)>1)>1):
            raise('fitting is currently only supported for 1D data')

    if(np.sum(np.isfinite(data))<4):
        print('ROBUSTMEAN:INSUFFICIENTDATA Less than 4 data points!')
        finalMean = np.nanmean(data,dim)
        stdSample = np.nanstd(data,[],dim)
        inlierIdx = matlab_find(np.isfinite(data))
        outlierIdx = []

        return finalMean,stdSample

    magicNumber2 = 1.4826**2

    dataSize = np.shape(data)
    reducedDataSize = np.array(dataSize)
    reducedDataSize[dim] = 1
    blowUpDataSize = dataSize/reducedDataSize
    realDimensions = len(matlab_find(np.array(dataSize)>1))
    if(fit):
        def func(x):
            return np.median(np.abs(data-x))
        medianData = fmin(func,np.median(data))
    else:
        medianData = np.nanmedian(data,dim)
    
    res2 = (data-medianData*np.ones(np.int32(blowUpDataSize)))**2
    eps = 2.2204e-16
    medRes2 = max([np.nanmedian(res2,dim),eps])

    testValue = res2/((magicNumber2*medRes2)*np.ones(np.int32(blowUpDataSize)))

    if(realDimensions == 1):
        inlierIdx = testValue<=k**2
        outlierIdx = testValue>k**2

        nInlier = np.sum(inlierIdx==True)
        if(nInlier>4):
            stdSample = np.sqrt(np.sum(res2[inlierIdx])/(nInlier-4))
        else:
            stdSample = np.nan
        finalMean = np.mean(data[0][inlierIdx[0]])
    else:
        inlierIdx = testValue<=k**2
        outlierIdx = testValue>k**2

        res2[outlierIdx] = np.nan
        nInlier = np.sum(np.logical_not(np.isnan(res2)),dim)

        goodIdx = np.sum(np.isfinite(res2),dim) > 4
        stdSample = np.nan*np.ones(np.shape(goodIdx))
        stdSample[goodIdx] = np.sqrt(np.nansum(res2[goodIdx],dim)/(nInlier[goodIdx]-4))

        data[outlierIdx] = np.nan
        finalMean = np.nanmean(data,dim)

    return finalMean,stdSample

def matlab_find(condition):
    return np.nonzero(condition)

def normalizstack(in_stack, p):
    size_in = np.shape(in_stack)
    midp = np.int32(ext_round((len(p['xrange'])+1)/2))

    xr = p['xrange'][(midp-4):(midp+3)]
    yr = p['yrange'][(midp-4):(midp+3)]

    if p['beadfilterf0']:
        out_stack = 0*in_stack+np.nan

        for k in range(size_in[3]):
            imh = in_stack[xr,yr,p['framerange'],k]
            nm = np.nanmean(imh)
            if(nm>0):
                out_stack[:,:,:,k] = in_stack[:,:,:,k]/nm

    else:
        inh = in_stack
        out_stack = 0*in_stack+np.nan
        for iter in range(4):
            meanim = np.nanmean(inh,3)

            for k in range(size_in[3]):
                imh = inh[:,:,:,k]
                if(np.all(np.isnan(imh))):
                    continue
                
                imhx=imh[xr,:,:]
                imhxy=imhx[:,yr,:]
                ims = imhxy[:,:,np.int32(p['framerange'])]
                
                meanimx=meanim[xr,:,:]
                meanimxy=meanimx[:,yr,:]
                meanims = meanimxy[:,:,np.int32(p['framerange'])]
                
                isn = np.logical_or(np.isnan(ims),np.isnan(meanims))
                intcutoff = meanims>np.quantile(meanims,0.75)
                indg = np.logical_and(np.logical_not(isn),intcutoff)
                ratio = ims[indg]/meanims[indg]

                factor = np.nanmedian(ratio)
                if factor>0:
                    out_stack[:,:,:,k] = imh/factor

            inh = out_stack

    return out_stack

def aligntoref(avim,imina, zshiftf0,p):
    xn = np.arange(1,np.size(imina,0) + 1)
    yn = np.arange(1,np.size(imina,1) + 1)
    zn = np.arange(1,np.size(imina,2) + 1)

    smallim = np.zeros([len(p['xrange']),len(p['yrange']),len(p['framerange']),np.size(imina,3)])
    for k in range(0,np.size(imina,3)):
        frh = ext_round(np.array(p['framerange'].astype(Decimal))-zshiftf0[k]).astype(int)
        try:
            smallim[:,:,:,k]=imina[:,:,frh,k][p['xrange'],:,:][:,p['yrange'],:]
        except:
            pass

    Xq,Yq,Zq = np.meshgrid(yn,xn,zn)
    refim=avim[:,:,p['framerange'].astype(int)][:,p['yrange'].astype(int),:][p['xrange'].astype(int),:,:]
    numbeads=np.size(imina,3)
    simin = np.shape(imina)
    shiftedstack=np.zeros([simin[0],simin[1],simin[2],numbeads])+np.nan

    shift = np.zeros([numbeads,3])
    cc = np.zeros(numbeads)

    for k in range(numbeads):
        p['status'] = 'calculate shift of individual PSFs: '+str(k)+' of '+str(numbeads)
        goodframes=np.nansum(np.nansum(smallim[:,:,:,k],0),0)>0
        if(p['alignz']):
            shift[k,:],cc[k] = get3Dcorrshift(refim[:,:,goodframes].copy(),smallim[:,:,goodframes,k].copy())
        else:
            if(np.any(goodframes)):
                print('[shift(k,:),cc(k)]=get2Dcorrshift(refim(:,:,goodframes),smallim(:,:,goodframes,k))')
            else:
                shift[k,:]=[0,0,0]
                cc[k]=np.nan

        b3 = bsarray(imina[:,:,:,k])
        shiftedh=interp3_0(b3,Xq-shift[k,1],Yq-shift[k,0],Zq-shift[k,2]-np.float64(zshiftf0[k]),0)
        shiftedstack[:, :, :, k] = shiftedh

    return shiftedstack,shift,cc