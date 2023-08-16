from .ext_round import *
from .interp3_0 import *
from .registerPSF3D_so import *
from .bsarray import *
from .mlefit import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def get_stackcal_so(beads, p):
    # set flags
    if (str.find(p['modality'],'astig')>=0):
        isastig = True
    else:
        isastig = False
    
    if (isastig and (str.find(p['zcorr'],'astig')>=0)):
        alignzastig = True
    else:
        alignzastig = False
    
    if (str.find(p['zcorr'],'corr')>=0):
        zcorr = True
    else:
        zcorr = False

    fig_list = []
    sstack=np.shape(beads[0]['stack']['image']) # image size
    halfstoreframes=ext_round((sstack[2] - 1) / 2)

    if(isastig):
        dframe = np.zeros(len(beads))
        for B in range(len(beads)-1,-1,-1):
            if halfstoreframes < len(beads[B]['stack']['framerange']):
                dframe[B] = beads[B]['stack']['framerange'][int(halfstoreframes)] - beads[B]['f0']
            else:
                dframe[B] = np.nan
                    
        for i in range(len(dframe)):
            if(abs(dframe[i]-np.nanmedian(dframe))>10 or np.isnan(dframe[i])):
                beads.pop(i)

        psfx = np.zeros(len(beads))
        psfy = np.zeros(len(beads))
        for i in range(len(beads)):
            psfx[i] = beads[i]['psfx0']
            psfy[i] = beads[i]['psfy0']

        dpsfx = (psfx-np.median(psfx[~(np.isnan(psfx))]))*10
        dpsfy = (psfy-np.median(psfy[~(np.isnan(psfy))]))*10
    else:
        dframe = 0
        dpsfx = 0
        dpsfy = 0

    allstacks = np.zeros([sstack[0], sstack[1], sstack[2], len(beads)]) + np.nan  # fushuang

    goodvs = np.zeros(len(beads))
    for B in range(len(beads)-1,-1,-1):
        stackh = beads[B]['stack']['image']
        allstacks[:,:,0:np.size(stackh,2),B] = stackh
        stackh = allstacks[:,:,:,B]
        goodvs[B] = np.sum(~np.isnan(stackh))/np.size(stackh)

    mstack=np.nanmean(allstacks,3)
    mstacks=np.reshape(mstack,[np.size(mstack),1],'F')[2:-2]
    mstack=mstack-np.nanmin(mstacks)
    mstack=mstack/np.nansum(mstack)
    
    dstack = np.zeros(len(beads))
    for k in range(len(beads)-1,-1,-1):
        stackh=(allstacks[:,:,:,k])
        stackh=stackh-np.nanmin(stackh)
        stackh=stackh/np.nansum(stackh)
        dstack[k]=np.sum((stackh-mstack)**2)
    dstack=dstack/np.mean(dstack)

    devs=(dpsfx**2+dpsfy**2+dstack)/goodvs

    if(zcorr):
        fw2 = ext_round((p['zcorrframes']-1)/2)
    else:
        fw2 = 2

    sortinddev = np.argsort(devs)

    if(alignzastig):
        zshift = dframe[sortinddev]-ext_round(np.median(dframe))
    else:
        zshift = []

    midrange = halfstoreframes+1-ext_round(np.median(dframe))
    # framerange = np.arange(np.max([0,midrange-fw2]),min([midrange+fw2,np.size(stackh,2)])+1) # matlab indexing, translate to python indexing later
    framerange = np.arange(np.max([0, midrange - fw2]), min([midrange + fw2, np.size(stackh, 2)]))


    p['status'] = 'calculate shift of individual PSFs' # update widget

    filenumber = np.zeros(len(beads))
    for i in range(len(beads)):
        filenumber[i] = beads[i]['filenumber']

    para = dict()
    para['sortind'] = sortinddev
    para['framerange'] = framerange
    para['alignz'] = zcorr
    para['zshiftf0'] = zshift
    para['beadfilterf0']=False

    corrPSF,shiftedstack,shift,beadgood = registerPSF3D_so(allstacks,para,[],filenumber[sortinddev])

    indgood=beadgood
    allrois=allstacks

    scorrPSF=np.shape(corrPSF)
    x=ext_round((scorrPSF[0]+1)/2)
    y=ext_round((scorrPSF[1]+1)/2)

    dRx=ext_round((p['ROIxy']-1)/2)
    if(('ROIz' not in p) or np.isnan(p['ROIz'])):
        p['ROIz']=np.size(corrPSF,2)
    
    dzroi=ext_round((p['ROIz']-1)/2)

    rangex = np.int32(np.arange(x-dRx-1,x+dRx))
    rangey = np.int32(np.arange(y-dRx-1,y+dRx))

    z = np.int32(midrange)
    rangez = np.arange(max((0,z-dzroi)),min((np.size(corrPSF,2),z+dzroi))).astype(np.int32)
    for z0reference in range(len(rangez)):
        if(rangez[z0reference]>=z):
            break

    corrPSFx = corrPSF[rangex,:,:]
    corrPSFxy = corrPSFx[:, rangey, :]
    centpsf = corrPSFxy[:, :, z-2:z+1]
    minPSF = np.nanmin(centpsf)
    corrPSFn = corrPSF - minPSF
    corrPSFnx = corrPSFn[rangex,:,:]
    corrPSFnxy = corrPSFnx[:,rangey,:]
    corrPSFnxyz = corrPSFnxy[:,:,z-2:z+1]
    intglobal = np.nanmean(np.nansum(np.nansum(corrPSFnxyz,0),0))
    corrPSFn = corrPSFn/intglobal
    
    shiftedstack = shiftedstack/intglobal
    corrPSFn[np.isnan(corrPSFn)]=0
    corrPSFn[corrPSFn<0]=0
    corrPSFnx = corrPSFn[rangex,:,:]
    corrPSFnxy = corrPSFnx[:,rangey,:]
    corrPSFs = corrPSFnxy[:,:,rangez]

    PSFgood = True

    lambdax = p['smoothxy']/p['cam_pixelsize_um'][0]/100000
    lambdaz = p['smoothz']/p['dz']*100
    lambda_arr = np.array((lambdax,lambdax,lambdaz))

    b3_0 = bsarray(corrPSFs,lambda_array=lambda_arr)

    zhd = np.arange(1,b3_0['dataSize'][2] + 1)
    dxxhd = 1

    XX,YY,ZZ = np.meshgrid(np.arange(1,b3_0['dataSize'][0] + 1,dxxhd),np.arange(1,b3_0['dataSize'][1]+1,dxxhd),zhd)
    corrPSFhd = interp3_0(b3_0,XX,YY,ZZ,0)
    coeff = psf2cspline_np(corrPSFhd)

    bspline = dict()
    bspline['bspline'] = b3_0
    cspline = dict()
    cspline['coeff'] = coeff
    cspline['z0'] = z0reference
    cspline['dz'] = p['dz']
    cspline['x0'] = dRx+1
    bspline['z0'] = ext_round((b3_0['dataSize'][2]+1)/2)
    bspline['dz'] = p['dz']
    splinefit = dict()
    splinefit['bspline'] = bspline
    p['z0'] = cspline['z0']

    splinefit['PSF'] = corrPSF
    splinefit['PSFsmooth'] = corrPSFhd
    splinefit['cspline'] = cspline

    if PSFgood:
        framerange0 = np.arange(p['fminmax'][0], p['fminmax'][1] + 1)
        halfroisizebig = (shiftedstack.shape[0] - 1) / 2
        ftest = int(z) - 1
        xt = int(x) - 1
        yt = int(y) - 1
        zpall = np.squeeze(shiftedstack[xt, yt, :, beadgood]).T
        zpall2 = np.squeeze(allrois[xt, yt,:, beadgood]).T
        xpall = np.squeeze(shiftedstack[:, yt, ftest, beadgood])
        xpall2 = np.squeeze(allrois[:, yt, ftest, beadgood])
        for k in range(0, zpall.shape[1]):
            zpall2[:, k]=zpall2[:, k] / np.nanmax(zpall2[:, k])
            xpall2[:, k]=xpall2[:, k] / np.nanmax(xpall2[:, k])

        zprofile = np.squeeze(corrPSFn[xt, yt,:])
        xprofile = np.squeeze(corrPSFn[:, yt, ftest])
        mpzhd = ext_round((corrPSFhd.shape[2] + 1) / 2 + 1)
        dzzz = ext_round((corrPSFhd.shape[2] - 1) / 2 + 1) - mpzhd
        dxxx = 0.1

        xxx = np.arange(1, b3_0['dataSize'][0] + dxxx, dxxx)
        zzzt = 0 * xxx + ftest
        xbs = interp3_0(b3_0, 0 * xxx + b3_0['dataSize'][0] / 2 + 0.5, xxx, zzzt)
        zzz = np.arange(1, b3_0['dataSize'][2] + dxxx, dxxx)
        xxxt = 0 * zzz + b3_0['dataSize'][0] / 2 + .5
        zbs = interp3_0(b3_0, xxxt, xxxt, zzz)

        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
        ax.set_xlabel('frames')
        ax.set_ylabel('normalized intensity')
        ax.set_xlim([np.min(framerange0), np.max(framerange0)])
        ax.set_axis_on()
        ax.set_title("Profile along z for x=0, y=0")
        ax.plot(framerange0, zpall[0:len(framerange0),:], color='cyan', label='individual PSFs')
        ax.plot(framerange0, zprofile[0:len(framerange0)], color='k', marker='*', label='average PSF')
        ax.plot(zzz + rangez[0] + framerange0[0] - 2, zbs, color='b', linewidth=2, label='smoothed spline')
        handles, labels = ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        ax.legend(handles, labels, loc='best')
        fig_list.append(['PSFz', figure])

        xrange = np.arange(-halfroisizebig, halfroisizebig+1)
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
        ax.set_xlabel('x (pixel)')
        ax.set_ylabel('normalized intensit')
        ax.set_axis_on()
        ax.set_title("Profile along x for y=0, z=0")
        ax.plot(xrange, xpall, color='cyan', label='individual PSFs')
        ax.plot(xrange, xprofile, color='k', marker='*', label='average PSF')
        ax.plot((xxx-(b3_0['dataSize'][0]+1)/2),xbs, color='b', linewidth=2, label='smoothed spline')
        handles, labels = ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        ax.legend(handles, labels, loc='best')
        fig_list.append(['PSFx', figure])

        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
        ax.set_xlabel('frame')
        ax.set_ylabel('zfit (nm)')
        ax.set_axis_on()
        testallrois = allrois[:,:,:, beadgood]
        testallrois[np.isnan(testallrois)] = 0
        testfit(testallrois,cspline['coeff'],p, [], ax)
        corrPSFfit = corrPSF / np.max(corrPSF) * np.max(testallrois)
        testfit(corrPSFfit,cspline['coeff'],p,['k',2], ax)
        #ax.legend()
        fig_list.append(['validate', figure])

    return splinefit, indgood, shift, fig_list


def testfit(teststack, coeff, p, linepar, ax):
    fitsize = min(p['ROIxy'], 21)
    d = ext_round((teststack.shape[0] - fitsize) / 2)
    range = np.arange(int(d), int(d + fitsize))
    if len(teststack.shape) > 3:
        numstack = teststack.shape[3]
        sx = teststack[range,:,:,:]
        sy = sx[:,range,:,:]
    else:
        numstack = 1
        sx = teststack[range, :, :]
        sy = sx[:, range, :]
    thispath = os.path.dirname(os.path.abspath(__file__))
    cpupath = thispath + '/CPUmleFit_LM.dll'
    gpupath = thispath + '/GPUmleFit_LM.dll'
    dll = mleFit(usecuda=0, cpu_path=cpupath)

    coeff_t = coeff.transpose(3, 2, 1, 0).copy()

    for k in np.arange(0, numstack):
        if '2D' in p['modality']:
            fitmode = 6
        else:
            fitmode = 5
        if len(teststack.shape) > 3:
            test_stack = np.squeeze(sy[:,:,:,k]).transpose(2, 1, 0).copy()
        else:
            test_stack = np.squeeze(sy).transpose(2, 1, 0).copy()
        P, CRLB, LL, coeff = dll.fit_coef(test_stack, fitmode, 100, coeff_t)

        z = np.arange(1, P.shape[1] + 1) - 1

        znm = (P[4, :] - p['z0']) * p['dz']
        if len(linepar) == 2:
            ax.plot(z, znm, color=linepar[0], linewidth=linepar[1])
        else:
            ax.plot(z, znm)
