using DynamicalSystems
using Catlab
using Catlab.CategoricalAlgebra
using AlgebraicDynamics.DiscDynam
using AlgebraicDynamics.Trajectory

""" In this example we approximate the Nose-Hoover ODE

dx/dt = y, dy/dt = -x + yz, dz/dt = 1 - y^2

(1) using the approximation tools in the DynamicalSystems.jl package
(2) using compositional Euler's method
"""

u0 = [0, 0.1, 0]        # initial condition
dt = .001               # time step
T  = 10                 # time to integrate over
N = Int(T/dt)

dotf(x) = [x[2], -x[1]]
dotg(x) = [x[1]*x[2], 1 - x[1]^2]


# DynamicalSystems.jl
ds = Systems.nosehoover(u0)
xs_ds = Systems.trajectory(ds, T, dt = dt)

# Compositional Euler's Method
f_eulers = Dynam(eulers(dotf), 2,[1,2], u0[1:2])
g_eulers = Dynam(eulers(dotg), 2,[1,2], u0[2:3])
cosp = Cospan(FinFunction([1, 2, 2, 3], 3), FinFunction([1,2,3], 3))

ns_eulers = AlgebraicDynamics.DiscDynam.compose(cosp)(f_eulers, g_eulers)

internal_eulers = AlgebraicDynamics.Trajectory.trajectory(ns_eulers, N, dt)
xs_eulers = mapslices(x -> get_exposed(ns_eulers, x), internal_eulers, dims = 2)