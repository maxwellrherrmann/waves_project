import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.collections as collections
from IPython.display import HTML

nx = 101; velocity = 1
x = np.linspace(0,1,nx)
dx = 1/nx; dt = dx/2


u = np.exp(-(10*(x-0.5))**2)
u_1 = np.exp(-(10*(x-0.5 - velocity*dt))**2)
u[0] = u[2]
u[-1] = u[-3]
plt.plot(x,u)
a = []
for t in range(0,500):
    a.append(np.copy(u))
    u_2 = np.copy(u_1)
    u_1 = np.copy(u)
    for i in range(1,nx-1):
        u[i] = 2*u_1[i] - u_2[i] + \
            (u_1[i+1] - 2*u_1[i] + u_1[i-1])*(velocity*dt/dx)**2
    u[0] = u[2]
    u[-1] = u_1[-2] + ((2-1)/(2+1))*(u_1[-1]-u[-2])

fig = plt.figure()
fig.set_dpi(100)
ax1 = fig.add_subplot(1,1,1)
k = 0

def animate(i):
    global k
    psi = a[k]
    k += 1
    ax1.clear()
    plt.plot(x,psi)
    #plt.grid(True)
    plt.ylim([0,1])
    plt.xlim([0.0,1.0])

anim = animation.FuncAnimation(fig,animate,frames=len(a)-2,interval=20)
