# # Combinatorial Threshold Linear Networks

using LinearAlgebra
using SparseArrays
using Catlab
using Catlab.Graphs
using OrdinaryDiffEq
using NonlinearSolve
using AlgebraicDynamics
using AlgebraicDynamics.ThresholdLinear

# ## Start testing the package

using Test
using Plots
using Catlab.Graphics

@test support([1,2,3.0]) == [1,2,3]
@test support([1,0,3.0]) == [1,3]
@test support([1,1e-15,1e-11]) == [1,3]

# ## Triangle Graph 

trig = @acset Graph begin
  V = 3
  E = 3
  src = [1,2,3]
  tgt = [2,3,1]
end

tri = TLNetwork(CTLNetwork(trig))
@test tri.W' == [ 0.0   -0.75  -1.5;
          -1.5    0.0   -0.75;
          -0.75  -1.5    0.0;
      ]
prob = ODEProblem(tri, [1,1/2,1/4], [0.0,40])
soln = solve(prob, alg=Tsit5())
plot(soln)

# test case that u1 = 0 still converges to attractor with support 123

prob = ODEProblem(tri, [0,1/2,1/4], [0.0,40])
soln = solve(prob, alg=Tsit5())
@test support(soln) == [1,2,3]
plot(soln)

# test cases for nonfull support

# Symmetric Edge Graph

barbell = CTLNetwork(@acset Graph begin
  V = 2
  E = 2
  src = [1,2]
  tgt = [2,1]
end)

# Adding an isolated vertex shouldn't affect the fixed points

bb3 = CTLNetwork(apex(coproduct([barbell.G, Graph(1)])))

TLNetwork(bb3).W
prob = ODEProblem(bb3, [1/2, 1/4, 1/8], (0.0,100))
soln = solve(prob, Tsit5())
plot(soln)
@test support(soln, 1e-14) == [1,2]

# Embedded subgraphs should have their fixed points persist.
# We add the extra vertex adjacent to both vertices in the barbell.

barbell_plus = CTLNetwork(@acset Graph begin
  V = 3
  E = 4
  src = [1,2,3,3]
  tgt = [2,1,1,2]
end)

TLNetwork(barbell_plus).W
prob = ODEProblem(barbell_plus, [1/2, 1/4, 1/8], (0.0,100))
soln = solve(prob, Tsit5())
@test support(soln, 1e-14) == [1,2]
plot(soln)

# When you direct sum graphs, you should cartesian product their fixed points.
# Two disjoint symmetric edges have fixed points on each.

barbell_pair = CTLNetwork(apex(coproduct(barbell.G, barbell.G)))
draw(barbell_pair.G)

# We first look for the attractor supported on the 3rd and 4th vertex (σ=[3,4]).

prob = ODEProblem(barbell_pair, [1/2, 1/4, 2/3, 4/5], (0,50))
soln = solve(prob, Tsit5())
@test support(soln, 1e-12)== [3,4]
plot(soln)

# With a different initial condition, we find another attractor this one corresponding to σ=[1,2].

prob = ODEProblem(barbell_pair, [1/2, 1/4, 2/50, 1/40], (0,50))
soln = solve(prob, Tsit5())
@test support(soln, 1e-12)== [1,2]
plot(soln)

# The graphs that have interesting structure have a lot of symetry so we try to make one with a product.

bt = apex(product(add_reflexives(barbell.G), trig))
tln = CTLNetwork(bt)
draw(tln.G)

# Now we look for an attractor.

prob = ODEProblem(tln, [1/2, 1/4, 2/3, 4/5, 1/10, 5/6], (0,150))
soln = solve(prob, Tsit5())
@test support(soln, 1e-5) == [1,2,4,5,6]
plot(soln)

# We can find another attractor by zeroing out some variables in the initial condition.

prob = ODEProblem(tln, [0, 1/4, 2/3, 0, 1/10, 0], (0,150))
soln = solve(prob, Tsit5())
plot(soln)

# Notice that even when you have only a singleton in the initial condition, you don't get a singleton support in the attractor.

prob = ODEProblem(tln, [0, 0, 2/3, 0, 0, 0], (0,150))
soln = solve(prob, Tsit5())
@test support(soln, 1e-5) == [1,2,3,5,6]
plot(soln)

# Because of symetry in the model, we can pick out a different attractor.

prob = ODEProblem(tln, [0, 2/3, 0, 0, 0, 0], (0,150))
soln = solve(prob, Tsit5())
@test support(soln, 1e-5) == [2,3,4,5,6]
plot(soln)

# ## Using Nonlinear Solvers to find fixed points
# NonlinearSolvers.jl lets us define the steady state of our system as our fixed point.
# We want unstable fixed points, so we can't use the DynamicSS problem type. 
# We have to use traditional rootfinders rather than an evolve to equilibirum approach.

prob = NonlinearProblem(tln, [0, 2/3, 0, 0, 0, 0])
fp = solve(prob)
fp.u

# Once we compute the fixed point, we can plug it in to the dynamics and simulate.
# This finds the corresponding ocscilatory attractor due to the numerical perturbations.
# We could converge to this attractor faster by adding our our perturbation to `fp.u`.

prob = ODEProblem(tln, fp.u, (0,150))
soln = solve(prob, Tsit5())
plot(soln)

# Because rootfinders are only guaranteed to find local minima of the objective,
# we start at a different initial guess and find a different attractor.

prob = NonlinearProblem(tln, [1/3, 0, 1/2, 1/2, 1/2, 1/2])
fp = solve(prob)

# We can plug in the fixed point and find the oscillatory attractor.

prob = ODEProblem(tln, fp.u, (0,150))
soln = solve(prob, Tsit5())
plot(soln)

# ## Induced Subgraphs Preserve Attractors
# When you take an induced subraph. You can restrict the dynamics onto that subgraph.

g = induced_subgraph(bt, [1,2,3])
tln = TLNetwork(CTLNetwork(g))
prob = NonlinearProblem(tln, [1/3, 0, 1/2])
fp = solve(prob)
support(fp.u)
prob = ODEProblem(tln, fp.u, (0,150))
soln = solve(prob, Tsit5())
plot(soln)

# Get the indicator function of a subset with respect to a graph.
# This should probably be a FinFunction.

indicator(g::AbstractGraph, σ::Vector{Int}) = map(vertices(g)) do v
  if v in σ
    return 1
  else
    return 0
  end
end

# The following two functions automate the analysis that we did above 
# 1. Restrict to a subgraph
# 2. Solve for a fixed point in the subgraph
# 3. Plug that solution in to the dynamics of the full system
# 4. Solve those dynamics and plot

function restriction_fixed_point(G::AbstractGraph, V::AbstractVector{Int})
  g = induced_subgraph(G, V)
  tln = TLNetwork(CTLNetwork(g))
  prob = NonlinearProblem(tln, ones(nv(g)) ./ nv(g))
  fp = solve(prob)
  σg = support(fp)
  σ  = V[σg]
  u = zeros(nv(G))
  map(σg) do v
    u[V[v]] = fp.u[v]
  end
  return σ, u
end

function restriction_simulation(G, V, tspan=(0,150.0))
  tln = CTLNetwork(G)
  σ, u₀ = restriction_fixed_point(G, V)
  prob = ODEProblem(tln, u₀, tspan)
  soln = solve(prob, Tsit5())
  plt = plot(soln)
  return σ, soln, plt
end

# ## Mining patterns in our product graph
# Let's take a look at our graph again

draw(bt)

# We can try finding an attractor from the triangle 2,4,6

σ₀, soln, plt = restriction_simulation(bt, [2,4,6]);
plt

# Here we can look at a tiny subgraph.

σ₀, soln, plt = restriction_simulation(bt, [1,2]);
plt

# There is a bigger support on 2,4,5,6

σ₀, soln, plt = restriction_simulation(bt, [2,4,5,6]);
plt

# Trying 1,2,4

σ₀, soln, plt = restriction_simulation(bt, [1, 2,4]);
plt

# Trying 1,2,3,5

σ₀, soln, plt = restriction_simulation(bt, [1,2,3,5]);
plt
