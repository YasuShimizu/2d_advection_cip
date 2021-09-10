import numpy as np

def fxfy_init(f,fx,fy,nx,ny,dx,dy):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            fx[i,j]=(f[i+1,j]-f[i-1,j])/(2.*dx)
            fy[i,j]=(f[i,j+1]-f[i,j-1])/(2.*dy)

    return fx,fy