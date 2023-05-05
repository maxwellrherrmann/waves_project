# Numerical Solution of the Wave Equation

The homogeneous wave equation is $\frac{\partial^2 u}{\partial t^2} = v^2\frac{\partial^2 u}{\partial \vec{x}^2}$, where $u$ is some solution and $v$ is the velocity of the wave.

We take a finite-difference approach to solving the wave equation in two-dimensions. 
Taking second-order centered-difference for both the temporal and spatial derivatives gives the wave equation in the form

```math
 \frac{u^{(n+1)}_{i,j} - 2u^{(n)}_{i,j} + u^{(n-1)}_{i,j}}{\left(\Delta t\right)^2} = v^2\frac{u^{(n)}_{i+1,j} + u^{(n)}_{i,j+1} - 4u^{(n)}_{i,j} + u^{(n)}_{i-1,j} + u^{(n)}_{i,j-1}}{\left(\Delta x\right)^2}
```

where the superscript $n$ denotes the time step and the subscripts $i,j$ denote the spatial location on the grid. 
At each point $(i,j)$, we solve the discretized wave equation for the value of the solution function $u$ at that point for the next time step $u_{i,j}^{(n+1)}$ via
```math
 u^{(n+1)}_{i,j} = 2u^{(n)}_{i,j} - u^{(n-1)}_{i,j} + \left(\frac{v \Delta t}{\Delta x}\right)^2\left[
u^{(n)}_{i+1,j} + u^{(n)}_{i,j+1} - 4u^{(n)}_{i,j} + u^{(n)}_{i-1,j} + u^{(n)}_{i,j-1}
\right].
```

## Stability and Error Analysis

The numerical stability of the numerical scheme presented here is most quickly evaluated through an examination of the Courant-Friedrichs-Lewy (CFL) condition. 
The CFL condition for the wave equation is the term in parentheses in the above equation, $\frac{v \Delta t}{\Delta x}$, which must be less than one for this formulation of the discrete wave equation. 
Satisfying the CFL condition is equivalent to imposing the constraint that the wave cannot move more than a single spatial step in one time step, which is necessary for maintaining numerical stability and correctly tracking the evolution of the wave through time.

The error in the numerical solution compared to the exact solution, also known as the truncation error, can be derived from Taylor series of the wave solution $u$ around a given point. 
Consider the Taylor series expansions of $u(x_0 + \Delta x,y)$:
```math
u(x_0+\Delta x,y) = u(x_0,y) + \left(\Delta x\right)u_x(x_0,y) + \frac{\left(\Delta x\right)^2}{2!}u_{xx}(x_0,y)
 + \frac{\left(\Delta x\right)^3}{3!}u_{xxx}(x_0,y) + \frac{\left(\Delta x\right)^4}{4!}u_{xxxx}(x_0,y) +\cdots, 
```
and $u(x_0 - \Delta x,y)$:
```math
 u(x_0-\Delta x,y) = u(x_0,y) - \left(\Delta x\right)u_x(x_0,y) + \frac{\left(\Delta x\right)^2}{2!}u_{xx}(x_0,y) 
 - \frac{\left(\Delta x\right)^3}{3!}u_{xxx}(x_0,y) + \frac{\left(\Delta x\right)^4}{4!}u_{xxxx}(x_0,y) + \cdots 
```
To get the second-order centered difference that we use, consider:
```math
 u(x_0+\Delta x,y) + u(x_0-\Delta x,y) = 2u(x_0,y) + \left(\Delta x\right)^2u_{xx}(x_0,y) + \frac{2\left(\Delta x\right)^4}{4!}u_{xxxx}(x_0,y) + \cdots 
 \Rightarrow u_{xx}(x_0,y) = \frac{u(x_0+\Delta x,y) + u(x_0-\Delta x,y) - 2u(x_0,y)}{\left(\Delta x\right)^2} + \mathcal{O}\left(\Delta x^2\right) 
```
For our case, in two-dimensions,
```math
 \Delta_h u - \Delta u = u^{(h)}_{xx} + u^{(h)}_{yy} - u_{xx} - u_{yy} = u_{xx} + \mathcal{O}\left(\Delta x^2\right) + u_{yy} + \mathcal{O}\left(\Delta x^2\right) - u_{xx} - u_{yy} = \mathcal{O}\left(\Delta x^2\right) 
 \partial_{t}^{(h)2} u - \partial_{t}^2 u = u^{(h)}_{tt}- u_{tt} = u_{tt} + \mathcal{O}\left(\Delta t^2\right) - u_{tt} = \mathcal{O}\left(\Delta t^2\right)\\
    \delta u &= \mathcal{O}\left(\Delta x^2\right) + \mathcal{O}\left(\Delta t^2\right)
```


## Dependencies

This code suite depends on the following Python modules
- `numpy`
- `matplotlib`
- `argparse`
- `csv`
- `os`

## Features

This project includes
- Neumann, Dirichlet, and absorbing boundary conditions for the simulation box
- N-slit, corner, and circular barriers for diffraction and scattering
- Animation of the simulation
- Extraction of diffraction patterns, both real-time and time-averaged

## How to use
Should be able to do something like
```python main.py --xbc neumann --ybc dirichlet ...```
etc. Definitely needs 
```python main.py --help```
and it might be nice to allow the user to run this from a fixed configuration file
```python main.py --config config.json```
