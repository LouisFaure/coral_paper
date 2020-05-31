#https://github.com/richardagalvez/Vortices-Python/blob/master/Vortex-FTLE.ipynb
#https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/FTLE-interp.html


import numpy as np
import math
from matplotlib import pyplot
import time
import sys
import numba

@numba.jit
def bilinear_interpolation(X, Y, f, x, y):
    """Returns the approximate value of f(x,y) using bilinear interpolation.
    
    Arguments
    ---------
    X, Y -- mesh grid.
    f -- the function f that should be an NxN matrix.
    x, y -- coordinates where to compute f(x,y)
    
    """
    
    N = np.shape(X[:,0])[0]

    dx, dy = X[0,1] - X[0,0], Y[1,0] - Y[0,0]
    x_start, y_start = X[0,0], Y[0,0] 
    
    
    i1, i2 = int ((x - x_start)/dx) , int((x - x_start)/dx) + 1
    j1, j2 = int ((y - y_start)/dy) , int((y - y_start)/dy) + 1
    
    
    # Take care of boundaries
    
    # 1. Right boundary
    
    if i1 >= N-1 and j1 <= N-1 and j1 >= 0:
        return f[j1, N-1]
    if i1 >= N-1 and j1 <= 0 :
        return f[0, N-1]
    if i1 >= N-1 and j1 >= N-1 :
        return f[N-1, N-1]
    
    # 2. Left boundary
    
    if i1 <= 0 and j1 <= N-1 and j1 >= 0:
        return f[j1, 0]
    if i1 <= 0 and j1 <= 0 :
        return f[0, 0]
    if i1 <= 0 and j1 >= N-1 :
        return f[N-1, 0]
    
    # 3. Top boundary
    
    if j1 >= N-1 and i1<=N-1 and i1>=0:
        return f[N-1, i1]
    if j1 >= N-1 and i1 <= 0 :
        return f[N-1, 0]
    
    # 3. Bottom boundary
    
    if j1 <= 0 and i1<=N-1 and i1>=0:
        return f[0, i1]
    if j1 <= 0 and i1 >= N-1 :
        return f[N-1, 0]
    

    x1, x2 = X[j1,i1], X[j2,i2]
    y1, y2 = Y[j1,i1], Y[j2,i2]
    
    f_interpolated = ( 1/(x2-x1)*1/(y2-y1) *
                      ( f[j1,i1]*(x2-x)*(y2-y) + f[j1,i2]*(x-x1)*(y2-y) 
                      + f[j2,i1]*(x2-x)*(y-y1) + f[j2,i2]*(x-x1)*(y-y1)) ) 
    
    return f_interpolated


@numba.jit 
def rk4(X, Y, x, y, f, h, dim):
    """Returns the approximate value of f(x,y) using bilinear interpolation.
    
    Arguments
    ---------
    X, Y -- mesh grid.
    x, y -- coordinates where to begin the evolution.
    f -- the function f that will be evolved.
    h -- the time step (usually referred to this as dt.)
    dim -- 0 for x and 1 for y.
    
    """
    
    k1 = h * bilinear_interpolation(X, Y, f, x, y)
    k2 = h * bilinear_interpolation(X, Y, f, x + 0.5 * h, y + 0.5 * k1)
    k3 = h * bilinear_interpolation(X, Y, f, x + 0.5 * h, y + 0.5 * k2)
    k4 = h * bilinear_interpolation(X, Y, f, x + h      , y + k3)
    
    if dim == 0:
        return x + 1./6 * k1 + 1./3 * k2 + 1./3 * k3 + 1./6 * k4
    elif dim == 1:
        return y + 1./6 * k1 + 1./3 * k2 + 1./3 * k3 + 1./6 * k4
    else:
        print('invalid dimension parameter passed to rk4, exiting')
        #sys.exit()


        
def test_trajectory(X, Y, u, v, i, j, integration_time, dt):
    """ Plots the trajectories of a few particles
    
    Arguments
    ---------
    X, Y -- mesh grid.
    i, j -- indices of the first particle on the mesh.
    integration_time -- the duration of the integration
    dt -- the finess of the time integration space.
    
    """

    size = 12
    N=X.shape[0]

    xs, ys = X[j,i], Y[j, i]

    traj_x , traj_y = np.zeros((N,N)), np.zeros((N,N))

    traj_x[j][i], traj_y[j][i] = xs, ys

    #print '(x0s, y0s) ->', xs, ys
    fig, ax = pyplot.subplots(dpi=150)
    ax.set_aspect('equal')
    print('begining trajectory calculation')
    
    colors = ['r','g','c'] # To plot more particles, add more colors
    
    for l, c in enumerate(colors):

        for k in range(0, int(integration_time/dt)):
        
            xs, ys = rk4(X, Y, xs, ys, u, dt, 0), rk4(X, Y, xs, ys, v, dt, 1) 
            traj_x[j][i] += xs
            traj_y[j][i] += ys
    
            #print '(xs, ys) ->', xs, ys
            pyplot.scatter(xs, ys, s=5, color = c)
            #print k*dt
        
        i += 1
        j += 1
    
        xs, ys = X[j,i], Y[j, i]
    
    pyplot.streamplot(X, Y, u, v, density=2, linewidth=1, arrowsize=1, arrowstyle='->', color='k')

    
    return None


@numba.jit
def integrate(x_y,integration_time, dt,X,Y,u,v):  
    xs = x_y[0]
    ys = x_y[1]
    tr_x = xs
    tr_y = ys
    for k in range(0, int(integration_time/dt)):
        xs, ys = rk4(X, Y, xs, ys, u, dt, 0), rk4(X, Y, xs, ys, v, dt, 1) 
        tr_x += xs
        tr_y += ys
    return([tr_x,tr_y])



def get_traj(X,Y, u, v, integration_time, dt,n_cores=10,verbose=True):
    """ Returns the FTLE particle trajectory, faster approach
    
    Arguments
    ---------
    x, y -- mesh grid coordinates
    dt -- integral time step
    """

    N = np.shape(X[:,1])[0]
    x = X[0,:]
    y = Y[:,0]
    
    xy=np.array(np.meshgrid(x, y)).T.reshape(-1,2).tolist()
    xy=list(map(np.array,xy))
    
    #res=np.zeros(xy.shape)
    res=np.apply_along_axis(integrate,1,xy,integration_time, dt,X,Y,u,v)
    #res=np.array(p_map(integrate, xy,disable=False))
    
    return res[:,0].reshape(N,N).T, res[:,1].T.reshape(N,N).T


def get_ftle(traj_x, traj_y, X, Y, integration_time):
    """ Returns the FTLE scalar field
    
    Mostly adapted from Steven's FTLE code (GitHub user stevenliuyi)
    
    Arguments
    ---------
    traj_x, traj_y -- The trajectories of the FTLE particles
    X, Y -- Meshgrid
    integration_time -- the duration of the integration time
    
    """
    
    N = np.shape(X[:,0])[0]
    ftle = np.zeros((N,N))
    
    for i in range(0,N):
        for j in range(0,N):
            # index 0:left, 1:right, 2:down, 3:up
            xt = np.zeros(4); yt = np.zeros(4)
            xo = np.zeros(2); yo = np.zeros(2)

            if (i==0):
                xt[0] = traj_x[j][i]; xt[1] = traj_x[j][i+1]
                yt[0] = traj_y[j][i]; yt[1] = traj_y[j][i+1]
                xo[0] = X[j][i];      xo[1] = X[j][i+1]
            elif (i==N-1):
                xt[0] = traj_x[j][i-1]; xt[1] = traj_x[j][i]
                yt[0] = traj_y[j][i-1]; yt[1] = traj_y[j][i]
                xo[0] = X[j][i-1]; xo[1] = X[j][i]
            else:
                xt[0] = traj_x[j][i-1]; xt[1] = traj_x[j][i+1]
                yt[0] = traj_y[j][i-1]; yt[1] = traj_y[j][i+1]
                xo[0] = X[j][i-1]; xo[1] = X[j][i+1]

            if (j==0):
                xt[2] = traj_x[j][i]; xt[3] = traj_x[j+1][i]
                yt[2] = traj_y[j][i]; yt[3] = traj_y[j+1][i]
                yo[0] = Y[j][i]; yo[1] = Y[j+1][i]
            elif (j==N-1):
                xt[2] = traj_x[j-1][i]; xt[3] = traj_x[j][i]
                yt[2] = traj_y[j-1][i]; yt[3] = traj_y[j][i]
                yo[0] = Y[j-1][i]; yo[1] = Y[j][i]
            else:
                xt[2] = traj_x[j-1][i]; xt[3] = traj_x[j+1][i]
                yt[2] = traj_y[j-1][i]; yt[3] = traj_y[j+1][i]
                yo[0] = Y[j-1][i]; yo[1] = Y[j+1][i]
        
            lambdas = eigs(xt, yt, xo, yo)
            if np.isnan(lambdas).all():
                ftle[j][i] = float('nan')
            else:
                ftle[j][i] = .5*np.log(max(lambdas))/(integration_time)
    
    return ftle

def eigs(xt, yt, xo, yo):
    ftlemat = np.zeros((2,2))
    ftlemat[0][0] = (xt[1]-xt[0])/(xo[1]-xo[0])
    ftlemat[1][0] = (yt[1]-yt[0])/(xo[1]-xo[0])
    ftlemat[0][1] = (xt[3]-xt[2])/(yo[1]-yo[0])
    ftlemat[1][1] = (yt[3]-yt[2])/(yo[1]-yo[0])
    
    if (True in np.isnan(ftlemat)): 
        return('nan')
    
    ftlemat = np.dot(ftlemat.transpose(), ftlemat)
    w, v = np.linalg.eig(ftlemat)

    return w