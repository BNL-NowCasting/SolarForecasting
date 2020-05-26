#include <stdlib.h>

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )

void cumsum_float(float* x, const int ny, const int nx, float* res) {
    int i,j,offset;
    
    res[0]=x[0];
    for (i=1; i<nx; ++i) 
        res[i]=res[i-1]+x[i];
    offset=0;
    for (i=1; i<ny; ++i) {
        offset += nx;
        res[offset]=res[offset-nx]+x[offset];
    }

    for (i=1; i<ny; ++i) {
        offset=i*nx;
        for (j=1; j<nx; ++j) {
            ++offset;
            res[offset]=res[offset-nx]+res[offset-1]-res[offset-nx-1]+x[offset];
        }
    }

}

void cumsum_char(unsigned char* x, const int ny, const int nx, float* res) {
    int i,j,offset;
    
    res[0]=x[0];
    for (i=1; i<nx; ++i) 
        res[i]=res[i-1]+x[i];
    offset=0;
    for (i=1; i<ny; ++i) {
        offset += nx;
        res[offset]=res[offset-nx]+x[offset];
    }

    for (i=1; i<ny; ++i) {
        offset=i*nx;
        for (j=1; j<nx; ++j) {
            ++offset;
            res[offset]=res[offset-nx]+res[offset-1]-res[offset-nx-1]+x[offset];
        }
    }

}

void rolling_mean2_c(int N, float* x, unsigned char* flag, const int ny, const int nx, float* res) {
    int i,j,ix1,ix2,iy1,iy2,offset1,offset2, offset;
    int N2, N3, cnt;
    float *csum, *cumn;
    N2=N/2; N3=N-N2-1;
    
    csum= (float*) malloc(nx*ny*sizeof(float));
    cumn= (float*) malloc(nx*ny*sizeof(float));
    
    cumsum_float(x,ny,nx,csum);
    cumsum_char(flag,ny,nx,cumn);

    for (i=0; i<N2+2; ++i) {
        offset=i*nx;
        iy2=min(i+N3,ny-1);
        offset2=iy2*nx;
        for (j=0; j<N2+2; ++j) {
            ix2=min(j+N3,nx-1);
            cnt=(cumn[offset2+ix2]);
            if (cnt>=1)            
                res[offset++]=(csum[offset2+ix2])/cnt;
            else
                res[offset++]=NAN;            
        }
        for (j=N2+2; j<nx; ++j) {
            ix1=j-N2-1; ix2=min(j+N3,nx-1);
            cnt=cumn[offset2+ix2]-cumn[offset2+ix1];
            if (cnt>=1)
                res[offset++]=(csum[offset2+ix2]-csum[offset2+ix1])/cnt;
            else
                res[offset++]=NAN;    
        }
    }
    for (i=N2+2; i<ny; ++i) {
        offset=i*nx;
        iy1=i-N2-1; iy2=min(i+N3,ny-1);
        offset1=iy1*nx; offset2=iy2*nx;
        for (j=0; j<N2+2; ++j) {
            ix2=min(j+N3,nx-1);
            cnt=cumn[offset2+ix2]-cumn[offset1+ix2];
            if (cnt>=1)
                res[offset++]=(csum[offset2+ix2]-csum[offset1+ix2])/cnt;
            else
                res[offset++]=NAN;                
        }
    }
    for (i=N2+2; i<ny; ++i) {
        offset=i*nx+N2+2;
        iy1=i-N2-1; iy2=min(i+N3,ny-1);
        offset1=iy1*nx; offset2=iy2*nx;
        for (j=N2+2; j<nx; ++j) {
            ix1=j-N2-1; ix2=min(j+N3,nx-1);
            cnt=(cumn[offset1+ix1]+cumn[offset2+ix2]-cumn[offset1+ix2]-cumn[offset2+ix1]);
            if (cnt>=1)            
                res[offset++]=(csum[offset1+ix1]+csum[offset2+ix2]-csum[offset1+ix2]-csum[offset2+ix1])/cnt;
            else
                res[offset++]=NAN;
        }
    }
    free(csum);
    free(cumn);

}

