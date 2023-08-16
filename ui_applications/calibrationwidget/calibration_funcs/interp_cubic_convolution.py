from scipy.interpolate import *
import numpy as np
import numba

def interp3_cubic_convolution(V,k):
    """
    matlab function interp3(V,k,'cubic'), cubic convolution interpolation
    :param V: sample data, (n1,n2,n3) tensor
    :param k: refinement factor, this results in 2^k-1 interpolated points between sample values.
    :return: interpolated values, (2**k*(n1-1)+1,2**k*(n1-1)+1,2**k*(n1-1)+1) tensor
    """
    n1 = np.size(V,0)
    n2 = np.size(V,1)
    n3 = np.size(V,2)

    # first interpolate each n1*n2 matrix
    Vqt1 = np.zeros([2**k*(n1-1)+1,2**k*(n2-1)+1,n3])
    for i in range(0,n3):
        Vqt1[:,:,i] = interp2_cubic_convolution(V[:,:,i], k)
    
    # reshape the intepolated (2**k*(n1-1)+1,2**k*(n1-1)+1,n3) data to be ((2**k*(n1-1)+1)*(2**k*(n1-1)+1),n3) matrix
    Vqt2 = np.reshape(Vqt1,[(2**k*(n1-1)+1)*(2**k*(n2-1)+1),n3])

    # interpolate each row of the reshaped matrix
    Vqt3 = interp1_cubic_convolution(Vqt2.transpose(),k).transpose()

    # reshape the interpolated matrix for output
    Vq = np.reshape(Vqt3,[2**k*(n1-1)+1,2**k*(n2-1)+1,2**k*(n3-1)+1])
    return Vq

def interp1_cubic_convolution(V,k):
    """
    matlab function interp1(V,k,'cubic'), cubic convolution interpolation for vector
    :param V: sample data, (n,m) size, vector or matrix, when V is matrix, interpolate for each column vector
    :param k: refinement factor, this results in 2^k-1 interpolated points between sample values.
    :return: interpolated values, (2**k*(n-1)+1,m) 
    """
    # check the shape of input data, transpose the row vector
    if(np.size(V,0)==1):
        V = V.transpose()
    if(len(np.shape(V))==1):
        V = np.array([V]).transpose()

    m = np.size(V,1)
    n = np.size(V,0)
    a = -0.5
    h = 2**(-k)

    retV = interp1_cubic_convolution_inner(V,k,m,n,a,h)

    return retV

@numba.jit(nopython=True)
def interp1_cubic_convolution_inner(V,k,m,n,a,h):
    retV = np.zeros((2**k*(n-1)+1,m))
    Vext = np.zeros((n+2,m))

    # interpolate rach column vector
    for iii in range(m):
        # extent the sample data
        Vext[0,iii] = V[2,iii]-3*V[1,iii]+3*V[0,iii]
        Vext[1:(n+1),iii] = V[:,iii]
        Vext[n+1,iii] = 3*V[n-1,iii]-3*V[n-2,iii]+V[n-3,iii]

        # cubic convolution
        for ii in range(n-1):
            for jj in range(2**k+1):
                s = jj*h
                K1 = a*s**3-2*a*s**2+a*s
                K2 = (a+2)*s**3-(a+3)*s**2+1
                K3 = -(a+2)*s**3+(2*a+3)*s**2-a*s
                K4 = -a*s**3+a*s**2
                K = np.array([K1,K2,K3,K4])
                retV[ii*2**k+jj,iii] = np.vdot(K,Vext[ii:(ii+4),iii])

    return retV

def interp2_cubic_convolution(V,k):
    """
    matlab function interp2(V,k,'cubic'), cubic convolution interpolation for matrix
    :param V: sample data, (n1,n2) size matrix
    :param k: refinement factor, this results in 2^k-1 interpolated points between sample values.
    :return: interpolated values, (2**k*(n1-1)+1,2**k*(n2-1)+1) 
    """
    # only accept matrix
    if(len(np.shape(V))!=2):
        return []

    n1 = np.size(V,0)
    n2 = np.size(V,1)
    a  = -0.5
    h = 2**(-k)

    retV = interp2_cubic_convolution_inner(V,k,n1,n2,a,h)

    return retV

@numba.jit(nopython=True)
def interp2_cubic_convolution_inner(V,k,n1,n2,a,h):
    retV = np.zeros((2**k*(n1-1)+1,2**k*(n2-1)+1))

    # extend sample data
    Vu = np.zeros((1,n2))
    for kk in range(n2):
        Vu[0,kk] = 3*V[0,kk] - 3*V[1,kk] + V[2,kk]

    Vb = np.zeros((1,n2))
    for kk in range(n2):
        Vb[0,kk] = 3*V[n1-1,kk] - 3*V[n1-2,kk] + V[n1-3,kk]

    Vl = np.zeros((n1,1))
    for kk in range(n1):
        Vl[kk,0] = 3*V[kk,0] - 3*V[kk,1] + V[kk,2]

    Vr = np.zeros((n1,1))
    for kk in range(n1):
        Vr[kk,0] = 3*V[kk,n2-1] - 3*V[kk,n2-2] + V[kk,n2-3]

    Vext = np.zeros((n1+2,n2+2))
    Vext[0,0] = 3*Vl[0,0]-3*Vl[1,0]+Vl[2,0]
    Vext[0,1:n2+1] = Vu[0,:]
    Vext[0,n2+1] = 3*Vr[0,0]-3*Vr[1,0]+Vr[2,0]
    Vext[1:n1+1,0] = Vl[:,0]
    Vext[1:n1+1,1:n2+1] = V
    Vext[1:n1+1,n2+1] = Vr[:,0]
    Vext[n1+1,0] = 3*Vl[n1-1,0]-3*Vl[n1-2,0]+Vl[n1-3,0]
    Vext[n1+1,1:n2+1] = Vb
    Vext[n1+1,n2+1] = 3*Vr[n1-1,0]-3*Vr[n1-2,0]+Vr[n1-3,0]

    # cubic convolution interpolation
    for ii1 in range(n1-1):
        for ii2 in range(n2-1):
            for jj1 in range(2**k+1):
                s1 = jj1*h
            
                K11 = a*s1**3-2*a*s1**2+a*s1
                K12 = (a+2)*s1**3-(a+3)*s1**2+1
                K13 = -(a+2)*s1**3+(2*a+3)*s1**2-a*s1
                K14 = -a*s1**3+a*s1**2
                K1 = np.array([K11,K12,K13,K14])
                for jj2 in range(2**k+1):
                    s2 = jj2*h
                    
                    K21 = a*s2**3-2*a*s2**2+a*s2
                    K22 = (a+2)*s2**3-(a+3)*s2**2+1
                    K23 = -(a+2)*s2**3+(2*a+3)*s2**2-a*s2
                    K24 = -a*s2**3+a*s2**2
                    K2 = np.array([K21,K22,K23,K24])

                    retV[ii1*2**k+jj1,ii2*2**k+jj2] = np.vdot(np.dot(K1,Vext[ii1:(ii1+4),ii2:(ii2+4)]),K2)

    return retV

def interp2_cubic_convolution_grid(V, pos, dx, w):
    # only accept matrix
    if(len(np.shape(V))!=2):
        return []
    
    if(len(pos)!=2):
        return []

    n1 = np.size(V,0)
    n2 = np.size(V,1)
    a  = -0.5

    retV = interp2_cubic_convolution_grid_inner(V, pos, dx, w, n1, n2, a)

    return retV

@numba.jit(nopython=True)
def interp2_cubic_convolution_grid_inner(V, pos, dx, w, n1, n2, a):
    # extend sample data
    Vu = np.zeros((1,n2))
    for kk in range(n2):
        Vu[0,kk] = 3*V[0,kk] - 3*V[1,kk] + V[2,kk]

    Vb = np.zeros((1,n2))
    for kk in range(n2):
        Vb[0,kk] = 3*V[n1-1,kk] - 3*V[n1-2,kk] + V[n1-3,kk]

    Vl = np.zeros((n1,1))
    for kk in range(n1):
        Vl[kk,0] = 3*V[kk,0] - 3*V[kk,1] + V[kk,2]

    Vr = np.zeros((n1,1))
    for kk in range(n1):
        Vr[kk,0] = 3*V[kk,n2-1] - 3*V[kk,n2-2] + V[kk,n2-3]

    Vext = np.zeros((n1+2,n2+2))
    Vext[0,0] = 3*Vl[0,0]-3*Vl[1,0]+Vl[2,0]
    Vext[0,1:n2+1] = Vu[0,:]
    Vext[0,n2+1] = 3*Vr[0,0]-3*Vr[1,0]+Vr[2,0]
    Vext[1:n1+1,0] = Vl[:,0]
    Vext[1:n1+1,1:n2+1] = V
    Vext[1:n1+1,n2+1] = Vr[:,0]
    Vext[n1+1,0] = 3*Vl[n1-1,0]-3*Vl[n1-2,0]+Vl[n1-3,0]
    Vext[n1+1,1:n2+1] = Vb
    Vext[n1+1,n2+1] = 3*Vr[n1-1,0]-3*Vr[n1-2,0]+Vr[n1-3,0]

    wn = int(w/dx)
    retV = np.zeros((2*wn+1,2*wn+1))

    # cubic convolution interpolation
    for ii1 in range(-w,w):
        for ii2 in range(-w,w):
            for jj1 in range(wn+1):
                s1 = jj1*dx
            
                K11 = a*s1**3-2*a*s1**2+a*s1
                K12 = (a+2)*s1**3-(a+3)*s1**2+1
                K13 = -(a+2)*s1**3+(2*a+3)*s1**2-a*s1
                K14 = -a*s1**3+a*s1**2
                K1 = np.array([K11,K12,K13,K14])
                for jj2 in range(wn+1):
                    s2 = jj2*dx
                    
                    K21 = a*s2**3-2*a*s2**2+a*s2
                    K22 = (a+2)*s2**3-(a+3)*s2**2+1
                    K23 = -(a+2)*s2**3+(2*a+3)*s2**2-a*s2
                    K24 = -a*s2**3+a*s2**2
                    K2 = np.array([K21,K22,K23,K24])

                    retV[(ii1+w)*wn+jj1,(ii2+w)*wn+jj2] = np.vdot(np.dot(K1,Vext[(pos[0]+ii1):(pos[0]+ii1+4),(pos[1]+ii2):(pos[1]+ii2+4)]),K2)

    return retV

def interp1_cubic_convolution_grid(V,pos,dx,w):
    # check the shape of input data, transpose the row vector
    if(np.size(V,0)==1):
        V = V.transpose()
    if(len(np.shape(V))==1):
        V = np.array([V]).transpose()

    m = np.size(V,1)
    n = np.size(V,0)
    a = -0.5

    retV = interp1_cubic_convolution_grid_inner(V, pos, dx, w, m, n, a)

    return retV

@numba.jit(nopython=True)
def interp1_cubic_convolution_grid_inner(V, pos, dx, w, m, n, a):
    wn = np.int32(w/dx)
    retV = np.zeros((2*wn+1,m))

    Vext = np.zeros((n+2,m))

    # interpolate rach column vector
    for iii in range(m):
        # extent the sample data
        Vext[0,iii] = V[2,iii]-3*V[1,iii]+3*V[0,iii]
        Vext[1:(n+1),iii] = V[:,iii]
        Vext[n+1,iii] = 3*V[n-1,iii]-3*V[n-2,iii]+V[n-3,iii]

        # cubic convolution
        for ii in range(-w,w):
            for jj in range(wn+1):
                s = jj*dx
                K1 = a*s**3-2*a*s**2+a*s
                K2 = (a+2)*s**3-(a+3)*s**2+1
                K3 = -(a+2)*s**3+(2*a+3)*s**2-a*s
                K4 = -a*s**3+a*s**2
                K = np.array([K1,K2,K3,K4])
                retV[(ii+w)*wn+jj,iii] = np.vdot(K,Vext[(pos+ii):(pos+ii+4),iii])

    return retV
    
def interp3_cubic_convolution_grid(V,pos,dx,w):
    n1 = np.size(V,0)
    n2 = np.size(V,1)
    n3 = np.size(V,2)

    wn = np.int32(w/dx)
    # first interpolate each (2*wn+1)*(2*wn+1) matrix
    Vqt1 = np.zeros((2*wn+1,2*wn+1,2*w+1))
    for i in range(pos[2]-w,pos[2]+w+1):
        Vqt1[:,:,i-(pos[2]-w)] = interp2_cubic_convolution_grid(V[:,:,i],pos[0:2],dx,w)
    
    # reshape the intepolated (2*wn+1,2*wn,n3) data to be ((2*wn*(2*wn+1),n3) matrix
    Vqt2 = np.reshape(Vqt1,[(2*wn+1)*(2*wn+1),2*w+1])

    # interpolate each row of the reshaped matrix
    Vqt3 = interp1_cubic_convolution_grid(Vqt2.transpose(),w,dx,w).transpose()

    # reshape the interpolated matrix for output
    Vq = np.reshape(Vqt3,[2*wn+1,2*wn+1,2*wn+1])
    return Vq
