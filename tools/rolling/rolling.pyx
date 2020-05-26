import numpy as np
cimport numpy as np
cimport cython

cdef extern from "fast_rolling.c":
    void rolling_mean2_c(int N, float* x, unsigned char* flag, int ny, int nx, float* res) 

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cumsum_float(np.float32_t[:, ::1] x, np.float32_t[:,::1] res):
    cdef np.intp_t i,j
    res[0,0]=x[0,0]
    for i in range(1,x.shape[0]):
        res[i,0]=res[i-1,0]+x[i,0]
    for i in range(1,x.shape[1]):
        res[0,i]=res[0,i-1]+x[0,i]
    for i in range(1,x.shape[0]):
        for j in range(1,x.shape[1]):
            res[i,j]=res[i-1,j]+res[i,j-1]-res[i-1,j-1]+x[i,j]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cumsum_char(np.uint8_t[:, ::1] x, np.float32_t[:,::1] res):
    cdef np.intp_t i,j
    res[0,0]=x[0,0]
    for i in range(1,x.shape[0]):
        res[i,0]=res[i-1,0]+x[i,0]
    for i in range(1,x.shape[1]):
        res[0,i]=res[0,i-1]+x[0,i]
    for i in range(1,x.shape[0]):
        for j in range(1,x.shape[1]):
            res[i,j]=res[i-1,j]+res[i,j-1]-res[i-1,j-1]+x[i,j]

@cython.boundscheck(False)
@cython.wraparound(False)
def cumsum(x):
    if x.dtype != np.float32: 
        x=x.astype(np.float32)
    cdef np.intp_t  ny, nx
    ny = x.shape[0]
    nx = x.shape[1]
    cdef np.float32_t[:, ::1] res = np.empty((ny,nx),dtype=np.float32)

    cumsum_float(x,res)   
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rolling_mean2_cy(x, flag, int N):
    if x.dtype != np.float32:
        x=x.astype(np.float32)
    if flag.dtype != np.uint8:
        flag=flag.astype(np.uint8);

    cdef int N2=N//2; 
    cdef int N3=N-N2-1;

    cdef np.intp_t  ny, nx, i, j, iy1,iy2,ix1,ix2
    ny = x.shape[0]
    nx = x.shape[1]    
    cdef np.float32_t[:, ::1] csum = np.empty((ny,nx),dtype=np.float32)
    cumsum_float(x,csum)
    cdef np.float32_t[:, ::1] cumn = np.empty((ny,nx),dtype=np.float32)
    cumsum_char(flag,cumn)

    ret=np.empty((ny,nx),dtype=np.float32)
    cdef np.float32_t[:, ::1] res = ret
    
    for i in range(N2+2):
        iy2=min(i+N3,ny-1);
        for j in range(N2+2):
            ix2=min(j+N3,nx-1);
            res[i,j]=(csum[iy2,ix2])/(cumn[iy2,ix2])
        for j in range(N2+2,nx):
            ix1=j-N2-1; ix2=min(j+N3,nx-1);
            res[i,j]=(csum[iy2,ix2]-csum[iy2,ix1])/(cumn[iy2,ix2]-cumn[iy2,ix1])
    for i in range(N2+2,ny):
        iy1=i-N2-1; iy2=min(i+N3,ny-1);
        for j in range(N2+2):
            ix2=min(j+N3,nx-1);
            res[i,j]=(csum[iy2,ix2]-csum[iy1,ix2])/(cumn[iy2,ix2]-cumn[iy1,ix2])
    for i in range(N2+2,ny):
        iy1=i-N2-1; iy2=min(i+N3,ny-1);
        for j in range(N2+2,nx):
            ix1=j-N2-1; ix2=min(j+N3,nx-1);
            res[i,j]=(csum[iy1,ix1]+csum[iy2,ix2]-csum[iy1,ix2]-csum[iy2,ix1])/(cumn[iy1,ix1]+cumn[iy2,ix2]-cumn[iy1,ix2]-cumn[iy2,ix1])
    
    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
def rolling_mean2(np.float32_t[:,::1] x, np.uint8_t[:,::1] flag, int N):
    cdef np.intp_t  ny, nx
    ny = x.shape[0]
    nx = x.shape[1]    
    
    ret=np.empty((ny,nx),dtype=np.float32)
    cdef np.float32_t[:, ::1] res = ret
    rolling_mean2_c(N,&x[0,0],&flag[0,0],ny,nx,&res[0,0]); 
    
    return ret
