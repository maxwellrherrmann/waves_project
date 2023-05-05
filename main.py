import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import csv
import os


def neumann_BC_y(psi, f=0, g=0, dx=0.01):
    """
    Applies Neumann boundary conditions to the y-boundaries of the
    wavefunction array. The function takes in the wavefunction array, psi,
    and the values of the wavefunction at the boundaries, f and g, and the 
    grid spacing, dx. The function returns the wavefunction array with the
    Neumann boundary conditions applied to the y-boundaries.
    Arguments
    ----------
    psi : numpy.ndarray
    f : float or numpy.ndarray
    g : float or numpy.ndarray
    dx : float
    Returns
    -------
    psi : numpy.ndarray
    """
    psi[0, :] = psi[2, :] - 2 * dx * f
    psi[-1, :] = psi[-3, :] - 2 * dx * g
    return psi


def neumann_BC_x(psi, f=0, g=0, dx=0.01):
    """
    Applies Neumann boundary conditions to the x-boundaries of the
    wavefunction array. The function takes in the wavefunction array, psi,
    and the values of the derivative of the wavefunction at the boundaries,
    f and g, and the grid spacing, dx. The function returns the wavefunction
    array with the Neumann boundary conditions applied to the x-boundaries.
    Arguments
    ----------
    psi : numpy.ndarray
    f : float or numpy.ndarray
    g : float or numpy.ndarray
    dx : float
    Returns
    -------
    psi : numpy.ndarray
    """
    psi[:, 0] = psi[:, 2] - 2 * dx * f
    psi[:, -1] = psi[:, -3] - 2 * dx * g
    return psi


def dirichlet_BC_y(psi, f=0, g=0, dx=0.01):
    """
    Applies Dirichlet boundary conditions to the y-boundaries of the
    wavefunction array. The function takes in the wavefunction array, psi,
    and the values of the derivative of the wavefunction at the boundaries,
    f and g, and the grid spacing, dx. The function returns the wavefunction
    array with the Dirichlet boundary conditions applied to the y-boundaries.
    Arguments
    ----------
    psi : numpy.ndarray
    f : float or numpy.ndarray
    g : float or numpy.ndarray
    dx : float
    Returns
    -------
    psi : numpy.ndarray
    """
    psi[0, :] = f
    psi[-1, :] = g
    return psi


def dirichlet_BC_x(psi, f=0, g=0, dx=0.01):
    """
    Applies Dirichlet boundary conditions to the x-boundaries of the
    wavefunction array. The function takes in the wavefunction array, psi,
    and the values of the wavefunction at the boundaries, f and g, and the
    grid spacing, dx. The function returns the wavefunction array with the
    Dirichlet boundary conditions applied to the x-boundaries.
    Arguments
    ----------
    psi : numpy.ndarray
    f : float or numpy.ndarray
    g : float or numpy.ndarray
    dx : float
    Returns
    -------
    psi : numpy.ndarray
    """
    psi[:, 0] = f
    psi[:, -1] = g
    return psi


def absorbing_BC_x(psi, psi_prev, r):
    """
    Applies absorbing boundary conditions to the x-boundaries of the
    wavefunction array. The function takes in the wavefunction array, psi,
    the previous wavefunction array, psi_prev, and a constant r that determines
    the strength of the absorbing boundary conditions. The function returns the 
    wavefunction matrix with the absorbing boundary conditions applied to the 
    x-boundaries.
    Arguments
    ----------
    psi : numpy.ndarray
    psi_prev : numpy.ndarray
    r : float
    Returns
    -------
    psi : numpy.ndarray
    """
    psi[:, 0] = psi_prev[:, 1] + ((r - 1) / (r + 1)) * (psi_prev[:, 2] - psi[:, 1])
    psi[:, -1] = psi_prev[:, -2] + ((r - 1) / (r + 1)) * (psi_prev[:, -1] - psi[:, -2])
    return psi


def absorbing_BC_y(psi, psi_prev, r):
    """
    Applies absorbing boundary conditions to the y-boundaries of the
    wavefunction array. The function takes in the wavefunction array, psi,
    the previous wavefunction array, psi_prev, and a constant r that determines
    the strength of the absorbing boundary conditions. The function returns the
    wavefunction matrix with the absorbing boundary conditions applied to the
    y-boundaries.
    Arguments
    ----------
    psi : numpy.ndarray
    psi_prev : numpy.ndarray
    r : float
    Returns
    -------
    psi : numpy.ndarray
    """
    psi[0, :] = psi_prev[1, :] + ((r - 1) / (r + 1)) * (psi_prev[2, :] - psi[1, :])
    psi[-1, :] = psi_prev[-2, :] + ((r - 1) / (r + 1)) * (psi_prev[-1, :] - psi[-2, :])
    return psi


def x_boundary_conditions(psi, psi_prev=None, xtype='neumann', xf=0, xg=0, dx=0, r=3):
    """
    Applies boundary conditions to the x-boundaries of the wavefunction array.
    The function takes in the wavefunction array, psi, the previous wavefunction
    array, psi_prev, the type of boundary conditions, xtype, and the values of
    the wavefunction at the boundaries, xf and xg, and the grid spacing, dx. The
    function returns the wavefunction matrix with the boundary conditions
    applied to the x-boundaries.
    Arguments
    ----------
    psi : numpy.ndarray
    psi_prev : numpy.ndarray
    xtype : str
    xf : float or numpy.ndarray
    xg : float or numpy.ndarray
    dx : float
    r : float
    Returns
    -------
    psi : numpy.ndarray
    """
    if xtype == 'neumann':
        return neumann_BC_x(psi, xf, xg, dx)
    elif xtype == 'dirichlet':
        return dirichlet_BC_x(psi, xf, xg, dx)
    elif xtype == 'absorbing':
        return absorbing_BC_x(psi, psi_prev, r)
    raise Exception("xtype and ytype must be either 'neumann' or 'dirichlet' or 'absorbing'")


def y_boundary_conditions(psi, psi_prev=None, ytype='neumann', yf=0, yg=0, dy=0, r=3):
    """
    Applies boundary conditions to the y-boundaries of the wavefunction array.
    The function takes in the wavefunction array, psi, the previous wavefunction
    array, psi_prev, the type of boundary conditions, ytype, and the values of
    the wavefunction at the boundaries, yf and yg, and the grid spacing, dy. The
    function returns the wavefunction matrix with the boundary conditions
    applied to the y-boundaries.

    Arguments
    ----------
    psi : numpy.ndarray
    psi_prev : numpy.ndarray
    ytype : str
    yf : float or numpy.ndarray
    yg : float or numpy.ndarray
    dy : float
    r : float
    Returns
    -------
    psi : numpy.ndarray
    """
    if ytype == 'neumann':
        return neumann_BC_y(psi, yf, yg, dy)
    elif ytype == 'dirichlet':
        return dirichlet_BC_y(psi, yf, yg, dy)
    elif ytype == 'absorbing':
        return absorbing_BC_y(psi, psi_prev, r)
    raise Exception("xtype and ytype must be either 'neumann' or 'dirichlet' or 'absorbing'")


def n_slit_barrier(x, y, position, n_slits, slit_dims):
    """
    Applies a barrier with n slits to the wavefunction array. The function
    takes in the the position of the barrier, position, the number of slits,
    n_slits, and the dimensions of the slits, slit_dims. The function returns
    a mask of the same size as grid that represents the barrier.
    Arguments
    ----------
    x : numpy.ndarray
    y : numpy.ndarray
    position : float
    n_slits : int
    slit_dims : tuple (height, width)
    Returns
    -------
    psi : numpy.ndarray
    """
    if n_slits == 0:
        return np.ones_like(x)
    # horiz_mask puts zeros at the x-position of the slit
    horiz_mask = (position < x[0, :]).astype(int)
    horiz_mask *= (x[0, :] < position + slit_dims[0]).astype(int)

    # vert_mask uses products of heaviside functions to create evenly spaced gaps up the y-axis of the simulation
    vert_mask = np.heaviside((y - 1 / (n_slits + 1)) + slit_dims[1], 1) * \
                np.heaviside(-(y - 1 / (n_slits + 1)) + slit_dims[1], 1)
    for i in range(1, n_slits):
        vert_mask += np.heaviside((y - (i + 1) / (n_slits + 1)) + slit_dims[1], 1) * \
                     np.heaviside(-(y - (i + 1) / (n_slits + 1)) + slit_dims[1], 1)

    # logical_nots applied at the end to make sure we're returning zeros where there is a barrier and ones otherwise.
    return np.logical_not((np.logical_not(vert_mask) * horiz_mask)).astype(int)


def corner_barrier(x, y, position, corner_width):
    """
    Applies a corner barrier to the wavefunction array. The function
    takes in the position of the right-side of the barrier, position,
    and the width of the barrier, corner_width. The function returns
    a mask of the same size as grid that represents the barrier.
    Arguments
    ----------
    x : numpy.ndarray
    y : numpy.ndarray
    position : float
    corner_width : float
    Returns
    -------
    psi : numpy.ndarray
    """

    # horiz_mask puts zeros at the x-position of the slit
    horiz_mask = (position < x[0, :]).astype(int)
    horiz_mask *= (x[0, :] < position + corner_width).astype(int)

    # the heaviside function cuts off half the simulation
    vert_mask = np.heaviside(y - 0.5, 1)

    # logical_nots applied at the end to make sure we're returning zeros where there is a barrier and ones otherwise.
    return np.logical_not((np.logical_not(vert_mask) * horiz_mask)).astype(int)


def circle_barrier(x, y, center, r):
    """
    Applies a circle barrier to the wavefunction array. The function
    takes in the position of the barrier, center, and the radius of
    the barrier, r. The function returns a mask of the same size as
    grid that represents the barrier.
    Arguments
    ----------
    x : numpy.ndarray
    y : numpy.ndarray
    center: tuple
    Returns
    -------
    psi : numpy.ndarray
    """

    # Return 1's at the positions outside each circle
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 > r ** 2


if __name__ == "__main__":
    # Initiate parser for handling commandline arguments
    parser = argparse.ArgumentParser(
        prog='python3 main.py',
        description='This program simulates a 2D wave equation solution using the finite difference method. The program takes in a config file (examples in the configs/ folder) to generate the simulation.',
        epilog='Created by: Keegan Finger, Max Herrmann, Sam Liechty of CU Boulder')

    # Add arguments to the parser
    parser.add_argument('-c', '--config', required=True, help="Config file to be used.")
    parser.add_argument('-og', '--gifout', help="Name of output gif file.")
    parser.add_argument('-op', '--powerout', help="Name of output png of time-averaged power plot.")
    parser.add_argument('-n', '--nsteps', type=int, help="Number of time steps to be simulated.")
    parser.add_argument('--fps', type=int, help="Frames per second of animation.")
    parser.add_argument('--cmap', help="Colormap to be used for animation.")
    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print('Config file "{}" does not exist.'.format(args.config))
        exit(1)

    config_file = args.config
    output_gif_file = args.gifout
    output_power_file = args.powerout

    # Read config file
    with open(config_file, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        config = {}
        special_rows = ['centers', 'widths', 'frequencies', 'heights', 'barrier_centers', 'barrier_radii', 'slit_dims']
        for row in reader:
            if row[0] in special_rows:
                config[row[0]] = [float(x) for x in row[1:]]
            else:
                config[row[0]] = row[1]

    # Create constants with values from config or defaults if nothing is provided
    if args.nsteps:
        nsteps = args.nsteps
    elif 'nsteps' in config.keys():
        nsteps = int(config['nsteps'])
    else:
        nsteps = 100

    if args.cmap:
        cmap = args.cmap
    elif 'cmap' in config.keys():
        cmap = config['cmap']
    else:
        cmap = 'viridis'

    nx = int(config['nx'])
    ny = int(config['ny'])
    velocity = 1
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    grid_x, grid_y = np.meshgrid(x, y)
    dx = 1 / nx
    dy = 1 / ny
    dt = dx / 2

    # Initial condition options
    initial = config['initial']

    u_0 = np.zeros([nx, ny])
    u_1 = np.zeros_like(u_0)

    # creates a gaussian wave-packet with a velocity to the right
    if initial == 'gaussian':
        if 'center' in config.keys():
            c = float(config['center'])
        else:
            c = 0.5
        if 'width' in config.keys():
            s = float(config['width'])
        else:
            s = 0.1
        if 'frequency' in config.keys():
            w = float(config['frequency'])
        else:
            w = 100
        if 'height' in config.keys():
            I = float(config['height'])
        else:
            I = 2

        u_0 = I * np.exp(-(0.5 * (grid_x - c) / s) ** 2) * np.cos(w * grid_x)
        u_1 = I * np.exp(-(0.5 * (grid_x - c + velocity * dt) / s) ** 2) * np.cos(w * (grid_x + velocity * dt))

    # creates a standing wave originating at the left of the simulation
    elif initial == 'standing':
        if 'frequency' in config.keys():
            w = float(config['frequency'])
        else:
            w = 100

        if 'height' in config.keys():
            I = float(config['height'])
        else:
            I = 2

        u_0 = np.heaviside(-grid_x, 1)
        u_1 = np.cos(w * dt) * np.heaviside(velocity * dt - grid_x, 1)

    # creates droplets of the form of a circular gaussian wave packet with velocity pointing away from the center
    elif initial == 'droplet':
        if 'centers' in config.keys():
            centers = [(config['centers'][i], config['centers'][i + 1]) for i in range(0, len(config['centers']), 2)]
        else:
            centers = [(0.5, 0.5)]
        if 'widths' in config.keys():
            ss = config['widths']
        else:
            ss = [0.1] * len(centers) / 2
        if 'frequencies' in config.keys():
            ws = config['frequencies']
        else:
            ws = [100] * len(centers)
        if 'heights' in config.keys():
            Is = config['heights']
        else:
            Is = [2] * len(centers)

        for i, center in enumerate(centers):
            cx = center[0]
            cy = center[1]

            # .00001 added to ensure no divide by 0
            u_0 += Is[i] * np.exp(-0.5 * (((grid_x - cx) / ss[i]) ** 2 + ((grid_y - cy) / ss[i]) ** 2)) * \
                   np.cos(ws[i] * np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2))
            new_x = grid_x - cx + velocity * dt * (grid_x - cx) / np.sqrt(
                .00001 + (grid_x - cx) ** 2 + (grid_y - cy) ** 2)
            new_y = grid_y - cy + velocity * dt * (grid_y - cy) / np.sqrt(
                .00001 + (grid_x - cx) ** 2 + (grid_y - cy) ** 2)
            u_1 += Is[i] * np.exp(-0.5 * ((new_x / ss[i]) ** 2 + (new_y / ss[i]) ** 2)) * np.cos(
                ws[i] * np.sqrt(new_x ** 2 + new_y ** 2))

    # creates a gaussian droplet with a sinusoidal source
    elif initial == 'standing_drop':
        if 'xcenter' in config.keys():
            cx = float(config['xcenter'])
        else:
            cx = 0.5
        if 'ycenter' in config.keys():
            cy = float(config['ycenter'])
        else:
            cy = 0.5
        if 'height' in config.keys():
            I = float(config['height'])
        else:
            I = 2
        if 'frequency' in config.keys():
            w = float(config['frequency'])
        else:
            w = 100

        # .00001 added to ensure no divide by 0
        u_0 = I * np.heaviside(-np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2), 1)
        new_x = grid_x - cx + velocity * dt * (grid_x - cx) / np.sqrt(.00001 + (grid_x - cx) ** 2 + (grid_y - cy) ** 2)
        new_y = grid_y - cy + velocity * dt * (grid_y - cy) / np.sqrt(.00001 + (grid_x - cx) ** 2 + (grid_y - cy) ** 2)
        u_1 = I * np.cos(w * dt) * np.heaviside(-np.sqrt(new_x ** 2 + new_y ** 2), 1)

    # Initialize array

    u = np.array([u_0, u_1, u_1])

    if 'xboundary' in config.keys():
        xbc = config['xboundary']

    else:
        xbc = 'absorbing'

    if 'yboundary' in config.keys():
        ybc = config['yboundary']

    else:
        ybc = 'absorbing'

    if xbc == 'absorbing':
        if 'rx' in config.keys():
            rx = float(config['rx'])
        else:
            rx = 2
    else:
        rx = 2

    if ybc == 'absorbing':
        if 'ry' in config.keys():
            ry = float(config['ry'])
        else:
            ry = 2
    else:
        ry = 2

    # Apply inititial boundary conditions -- for simplicity this is always Neumann
    for i in range(3):
        u[i, :, :] = x_boundary_conditions(u[i, :, :], xtype='neumann')
        u[i, :, :] = y_boundary_conditions(u[i, :, :], ytype='neumann')

    # Main body algorithm

    # For keeping track of the wave at each timestep
    a = []

    # create barrier
    barrier = np.ones_like(u[0])
    if 'barrier' in config.keys():
        barrier_type = config['barrier']
    else:
        barrier_type = 'None'

    if barrier_type == 'nslit':

        if 'position' in config.keys():
            position = float(config['position'])
        else:
            position = 0.7

        if 'nslit' in config.keys():
            try:
                nslit = int(config['nslit'])
            except:
                print('nslit must be an integer. Defaulting to nslit=2')
                nslit = 2
        else:
            nslit = 2

        if 'slit_dims' in config.keys():
            slit_dims = config['slit_dims']
        else:
            slit_dims = (0.1, 0.02)

        # barrier = n_slit_barrier(grid_x, grid_y, 0.7, 3, (0.1, 0.02) )
        barrier = n_slit_barrier(grid_x, grid_y, position, nslit, slit_dims)

    elif barrier_type == 'corner':
        if 'position' in config.keys():
            position = float(config['position'])
        else:
            position = 0.7

        if 'corner_width' in config.keys():
            corner_width = float(config['corner_width'])
        else:
            corner_width = 0.1

        # barrier = corner_barrier(grid_x,grid_y,0.7,0.1)
        barrier = corner_barrier(grid_x, grid_y, position, corner_width)

    elif barrier_type == 'circles':
        if 'barrier_centers' in config.keys():
            barrier_centers = [(config['barrier_centers'][i], config['barrier_centers'][i + 1]) for i in
                               range(0, len(config['barrier_centers']), 2)]
        else:
            barrier_centers = [(.8, .5), (.8, .4), (.8, .6)]

        if 'barrier_radii' in config.keys():
            barrier_radii = config['barrier_radii']
        else:
            barrier_radii = [.05] * len(barrier_centers)

        for center, radius in zip(barrier_centers, barrier_radii):
            barrier *= circle_barrier(grid_x, grid_y, center, radius)

        try:
            assert (len(barrier_centers) == len(barrier_radii))
        except:
            print('barrier_centers and barrier_radii must correspond to a consistent number of circular barriers')
            exit(1)

    # Finite difference method for solving differential equations
    for t in range(0, nsteps):
        a.append(np.copy(u[0, :, :]))
        u[2, :, :] = u[1, :, :]
        u[1, :, :] = u[0, :, :]
        u[0, 1:nx - 1, 1:ny - 1] = 2 * u[1, 1:nx - 1, 1:ny - 1] - u[2, 1:nx - 1, 1:ny - 1] + \
                                   (u[1, 2:nx, 1:ny - 1] - 2 * u[1, 1:nx - 1, 1:ny - 1] + u[1, 0:nx - 2, 1:ny - 1] +
                                    u[1, 1:nx - 1, 2:ny] - 2 * u[1, 1:nx - 1, 1:ny - 1] + u[1, 1:nx - 1, 0:ny - 2]) * \
                                   (velocity * dt / dx) ** 2

        # apply boundary conditions after each iteration
        u[0, :, :] *= barrier
        u[0, :, :] = x_boundary_conditions(u[0, :, :], psi_prev=u[1, :, :], xtype=xbc, r=rx)
        u[0, :, :] = y_boundary_conditions(u[0, :, :], psi_prev=u[1, :, :], ytype=ybc, r=ry)

        # generate the source of standing options
        if initial == 'standing':
            u += I * np.cos(w * t * dt) * np.heaviside(-grid_x, 1)
        if initial == 'standing_drop':
            u += I * np.cos(w * t * dt) * np.heaviside(-np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2), 1)

    # settings to plot the figures
    fig1 = plt.figure()
    fig1.set_dpi(100)
    ax1 = fig1.add_subplot(1, 3, (1, 2))
    ax2 = fig1.add_subplot(1, 3, 3, sharey=ax1)
    psi = np.ones((nx, ny)) * float('nan')
    meshplot = ax1.pcolormesh(grid_x, grid_y, psi, clim=(-1, 1), cmap=cmap)
    power_dist, = ax2.plot(np.zeros_like(y), y, color="red", lw=2)
    ax1.set_xlim(left=0, right=1)
    ax1.set_ylim(bottom=0, top=1)
    ax2.set_xlim(left=0, right=4)
    ax2.set_xlabel('Power (arb. units)')
    ax1.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
    ax2.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
    k = 0

    # Plot the scalar wave
    def animate(i):
        global k
        try:
            psi = a[k]
        except IndexError:
            k = 0
            psi = a[k]
        meshplot.set_array(psi.ravel())
        power_dist.set_xdata(a[k][:, nx - 1] ** 2)
        k += 1

    # choose which plots to show and/or save
    if 'show' in config.keys() or output_gif_file:
        anim = animation.FuncAnimation(fig1, animate, frames=len(a) - 2, interval=10)

    if 'show' in config.keys():
        if config['show']:
            plt.show()

    if output_gif_file:
        anim.save(f'{output_gif_file}.gif', fps=30)

    if output_power_file:
        fig2 = plt.figure()
        ax3 = fig2.add_subplot(1, 1, 1)
        ax3.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)

        # loop through the simulation and find the non-zero wave amplitude on the right side of the simulation
        power = []
        for i in range(len(a)):
            if max((a[i][:, nx - 1]) ** 2) > 0:
                power.append(a[i][:, nx - 1] ** 2)
        # check if wave made it to the other side
        if power == []:
            print(
                'No power detected on far screen. Try increasing the number of steps. No output power plot generated.')
            exit(1)

        # average total power recieved
        ta_power = np.average(power, axis=0)

        ax3.plot(y, ta_power)
        plt.savefig(f'{output_power_file}.png')