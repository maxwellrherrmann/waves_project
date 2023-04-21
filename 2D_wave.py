import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.collections as collections
from IPython.display import HTML

def neumann_BC_y(psi,f=0,g=0,dx=0.01):
    psi[0,:] = psi[2,:] - 2*dx*f
    psi[-1,:] = psi[-3,:] - 2*dx*g
    return psi

def neumann_BC_x(psi,f=0,g=0,dx=0.01):
    psi[:,0] = psi[:,2] - 2*dx*f
    psi[:,-1] = psi[:,-3] - 2*dx*g
    return psi

def dirichlet_BC_y(psi,f=0,g=0,dx=0.01):
    psi[0,:] = f
    psi[-1,:] = g
    return psi

def dirichlet_BC_x(psi,f=0,g=0,dx=0.01):
    psi[:,0] = f
    psi[:,-1] = g
    return psi

def absorbing_BC_x(psi,psi_prev,r):
    psi[:,0] = psi_prev[:,1] + ((r-1)/(r+1))*(psi_prev[:,2]-psi[:,1])
    psi[:,-1] = psi_prev[:,-2] + ((r-1)/(r+1))*(psi_prev[:,-1]-psi[:,-2])
    return psi

def absorbing_BC_y(psi,psi_prev,r):
    psi[0,:] = psi_prev[1,:] + ((r-1)/(r+1))*(psi_prev[2,:]-psi[1,:])
    psi[-1,:] = psi_prev[-2,:] + ((r-1)/(r+1))*(psi_prev[-1,:]-psi[-2,:])
    return psi

def x_boundary_conditions(psi,psi_prev=None,xtype='n',xf=0,xg=0,dx=0,r=3):
    if xtype == 'n':
        return neumann_BC_x(psi,xf,xg,dx)
    elif xtype == 'd':
        return dirichlet_BC_x(psi,xf,xg,dx)
    elif xtype == 'a':
        return absorbing_BC_x(psi,psi_prev,r)
    raise Exception("xtype and ytype must be either 'n' or 'd' or 'a'")

def y_boundary_conditions(psi,psi_prev=None,ytype='n',yf=0,yg=0,dy=0,r=3):
    if ytype == 'n':
        return neumann_BC_y(psi,yf,yg,dy)
    elif ytype == 'd':
        return dirichlet_BC_y(psi,yf,yg,dy)
    elif ytype == 'a':
        return absorbing_BC_y(psi,psi_prev,r)
    raise Exception("xtype and ytype must be either 'n' or 'd'")

# Single Slit Barrier
single_slit_position = 0.65
single_slit_width = 0.05
single_slit_height = 0.2
single_slit_patch1 = patches.Rectangle((0.65,0),0.05,0.4,color='k')
single_slit_patch2 = patches.Rectangle((0.65,0.6),0.05,0.4,color='k')
ss_patches = collections.PatchCollection([single_slit_patch1,single_slit_patch2],color='k')
def single_slit(psi,x,y):
    if (single_slit_position <= x and \
        x <= single_slit_position + single_slit_width and\
        (y <= 0.5 - single_slit_height/2 or y >= 0.5 + single_slit_height/2)):
        return 0
    else:
        return psi

def single_slit_slow(psi=None,x=None,y=None):
    x_position = 0.65
    width = 0.05
    height = 0.2
    barrier = [patches.Rectangle((x_position,0),width,(1-height)/2),\
               patches.Rectangle((x_position,0.6),width,(1-height)/2)]
    patch_collection = collections.PatchCollection(barrier,color='k')
    if psi != None:
        for patch in barrier:
            if patch.contains_point((x,y)):
                return 0
            else:
                return psi
    else:
        return patch_collection

def double_slit_slow(psi=None,x=None,y=None):
    x_position = 0.65
    width = 0.05
    height = 0.2
    barrier = [patches.Rectangle((x_position,0),width,(0.5-height)/2),\
               patches.Rectangle((x_position,(0.5-height)/2+height),width,(1-height)/2-height/2),\
               patches.Rectangle((x_position,(1-height)/2-3*height/2), width, 1)]
    patch_collection = collections.PatchCollection(barrier,color='k')
    if psi != None:
        for patch in barrier:
            if patch.contains_point((x,y)):
                return 0
            else:
                return psi
    else:
        return patch_collection


nx = 101; ny = 101; velocity = 1
x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)
grid_x, grid_y = np.meshgrid(x,y)
dx = 1/nx; dy = 1/ny; dt = dx/2
u = np.exp(-(10*(grid_x-0.2))**2) * np.cos(100*grid_x)
u_1 = np.exp(-(10*(grid_x-0.2 + velocity*dt))**2) * np.cos(100*(grid_x + velocity*dt))
u = x_boundary_conditions(u,xtype='n')
u = y_boundary_conditions(u,ytype='n')
u_1 = x_boundary_conditions(u_1,xtype='n')
u_1 = y_boundary_conditions(u_1,ytype='n')
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.pcolor(grid_x,grid_y,u)
ax.add_collection(single_slit_slow())
plt.clim(-1,1)

a = []
for t in range(0,500):
    a.append(np.copy(u))
    u_2 = np.copy(u_1)
    u_1 = np.copy(u)
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            u[i,j] = 2*u_1[i,j] - u_2[i,j] + \
                (u_1[i+1,j] - 2*u_1[i,j] + u_1[i-1,j] +\
                u_1[i,j+1] - 2*u_1[i,j] + u_1[i,j-1])*(velocity*dt/dx)**2
            u[i,j] = single_slit(u[i,j],grid_x[i,j],grid_y[i,j])
    u = x_boundary_conditions(u,psi_prev=u_1,xtype='a',r=2)
    u = y_boundary_conditions(u,ytype='n')
    if t%100 == 0:
        print(t)




plt.style.use('dark_background')

fig = plt.figure()
fig.set_dpi(100)
ax1 = fig.add_subplot(1,1,1)
k = 0

def animate(i):
    global k
    psi = a[k]
    k += 1
    ax1.clear()
    plt.pcolor(grid_x,grid_y,psi)
    plt.clim(-1,1)
    ax1.add_collection(single_slit_slow())
    plt.ylim([0,1])
    plt.xlim([0.0,1.0])

anim = animation.FuncAnimation(fig,animate,frames=len(a)-2,interval=20)
anim.save('2D_wave.gif')
