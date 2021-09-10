import numpy as np
import copy
from numba import jit
@jit
def f_cal(f,fn,fx,fy,fxn,fyn,nx,ny,dx,dy,dt,u,v):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny): 
            xx=-u*dt;  yy=-v*dt
            isn=int(np.sign(u)); jsn=int(np.sign(v))
            if isn==0:
                isn=1
            if jsn==0:
                jsn=1
            im=i-isn; jm=j-jsn
            a1=((fx[im,j]+fx[i,j])*dx*isn-2.*(f[i,j]-f[im,j]))/(dx**3*isn)
            b1=((fy[i,jm]+fy[i,j])*dy*jsn-2.*(f[i,j]-f[i,jm]))/(dy**3*jsn)
            e1=(3.*(f[im,j]-f[i,j])+(fx[im,j]+2.*fx[i,j])*dx*isn)/dx**2
            f1=(3.*(f[i,jm]-f[i,j])+(fy[i,jm]+2.*fy[i,j])*dy*jsn)/dy**2
            tmp=f[i,j]-f[i,jm]-f[im,j]+f[im,jm]
            tmq1=fy[im,j]-fy[i,j]
            tmp2=fx[i,jm]-fx[i,j]
            c1=(-tmp-tmp2*dx*isn)/(dx**2*dy*jsn)
            d1=(-tmp-tmq1*dy*jsn)/(dx*dy**2*isn)
            g1=(-tmq1+c1*dx**2)/(dx*isn)
            fn[i,j]=((a1*xx+c1*yy+e1)*xx+g1*yy+fx[i,j])*xx \
              +((b1*yy+d1*xx+f1)*yy+fy[i,j])*yy+f[i,j]            
            fxn[i,j]=(3.*a1*xx+2.*(c1*yy+e1))*xx+(d1*yy+g1)*yy+fx[i,j]
            fyn[i,j]=(3.*b1*yy+2.*(d1*xx+f1))*yy+(c1*xx+g1)*xx+fy[i,j]
    return fn,fxn,fyn