```@meta
EditURL = "TLN.md"
```

# Threshold Linear Networks

These examples are based on the paper [Graph Rules for Recurrent
Neural Network Dynamics](https://www.ams.org/notices/202304/rnoti-p536.pdf) by Carina Curto and Katherine Morrison. 

````@example TLN
using LinearAlgebra
using SparseArrays
using Catlab
using Catlab.Graphs
using OrdinaryDiffEq
using NonlinearSolve
using AlgebraicDynamics
using AlgebraicDynamics.ThresholdLinear
using Test
using Plots
using Catlab.Graphics
````

## Triangle Graph

````@example TLN
trig = @acset Graph begin
  V = 3
  E = 3
  src = [1,2,3]
  tgt = [2,3,1]
end
````

````@example TLN
tri = TLNetwork(CTLNetwork(trig))
@test tri.W' == [ 0.0   -0.75  -1.5;
          -1.5    0.0   -0.75;
          -0.75  -1.5    0.0;
      ]
prob = ODEProblem(tri, [1,1/2,1/4], [0.0,40])
soln = solve(prob, alg=Tsit5())
plot(soln)
````

Test case that u1 = 0 still converges to attractor with support 123

````@example TLN
prob = ODEProblem(tri, [0,1/2,1/4], [0.0,40])
soln = solve(prob, alg=Tsit5())
plot(soln)
````

````@example TLN
support(soln)
````


## Symmetric Edge Graph

We want to build a test case for non-full support. We look to the symmetric edge graph.

````@example TLN
barbell = CTLNetwork(@acset Graph begin
  V = 2
  E = 2
  src = [1,2]
  tgt = [2,1]
end)
````

From the theory, we know that adding an isolated vertex shouldn't affect the 
fixed points. The isolated vertex has no activators, so it will decay to zero,
and it activates nothing, so it can't change those fixed points.

````@example TLN
bb3 = CTLNetwork(apex(coproduct([barbell.G, Graph(1)])))
prob = ODEProblem(bb3, [1/2, 1/4, 1/8], (0.0,100))
soln = solve(prob, Tsit5())
@test support(soln, 1e-14) == [1,2]
plot(soln)
````

Embedded subgraphs should have their fixed points persist.
We add the extra vertex adjacent to both vertices in the barbell.

````@example TLN
barbell_plus = CTLNetwork(@acset Graph begin
  V = 3
  E = 4
  src = [1,2,3,3]
  tgt = [2,1,1,2]
end)
TLNetwork(barbell_plus).W
````

Notice how symmetric this W matrix is.

````@example TLN
prob = ODEProblem(barbell_plus, [1/2, 1/4, 1/8], (0.0,100))
soln = solve(prob, Tsit5())
@test support(soln, 1e-14) == [1,2]
plot(soln)
````

The categorical coproduct of graphs gives you the direct sum of their adjacency matrices.
When you direct sum graphs, you should cartesian product their fixed points.
Two disjoint symmetric edges have fixed points on each.

````@example TLN
barbell_pair = CTLNetwork(apex(coproduct(barbell.G, barbell.G)))
draw(barbell_pair.G)
````

We first look for the attractor supported on the 3rd and 4th vertex (σ=[3,4]).

````@example TLN
prob = ODEProblem(barbell_pair, [1/2, 1/4, 2/3, 4/5], (0,50))
soln = solve(prob, Tsit5())
@test support(soln, 1e-12)== [3,4]
plot(soln)
````

With a different initial condition, we find another attractor this one corresponding to σ=[1,2].

````@example TLN
prob = ODEProblem(barbell_pair, [1/2, 1/4, 2/50, 1/40], (0,50))
soln = solve(prob, Tsit5())
@test support(soln, 1e-12)== [1,2]
plot(soln)
````

The graphs that have interesting structure have a lot of symmetry so we try to make one with a product.
Categorical products have symmetry because they work like the cartesian product of sets.

````@example TLN
bt = apex(product(add_reflexives(barbell.G), trig))
tln = CTLNetwork(bt)
draw(tln.G)
````

Now we look for an attractor.

````@example TLN
prob = ODEProblem(tln, [1/2, 1/4, 2/3, 4/5, 1/10, 5/6], (0,150))
soln = solve(prob, Tsit5())
@test support(soln, 1e-5) == [1,2,4,5,6]
plot(soln)
````

We can find another attractor by zeroing out some variables in the initial condition.

````@example TLN
prob = ODEProblem(tln, [0, 1/4, 2/3, 0, 1/10, 0], (0,150))
soln = solve(prob, Tsit5())
plot(soln)
````

Notice that even when you have only a singleton in the initial condition, you don't get a singleton support in the attractor.

````@example TLN
prob = ODEProblem(tln, [0, 0, 2/3, 0, 0, 0], (0,150))
soln = solve(prob, Tsit5())
# @test support(soln, 1e-5)  == [2,3,6]
plot(soln)
````

Because of symmetry in the model, we can pick out a different attractor.

````@example TLN
prob = ODEProblem(tln, [0, 2/3, 0, 0, 0, 0], (0,150))
soln = solve(prob, Tsit5())
# @test support(soln, 1e-5) == [2,3,4]
plot(soln)
````

## Using Nonlinear Solvers to find fixed points
NonlinearSolvers.jl lets us define the steady state of our system as our fixed point.
We want unstable fixed points, so we can't use the `DynamicSS` problem type provided by SciML.
We have to use traditional root finders rather than an evolve to equilibrium approach.

````@example TLN
prob = NonlinearProblem(tln, [0, 2/3, 0, 0, 0, 0])
fp = solve(prob)
fp.u
````

Once we compute the fixed point, we can plug it in to the dynamics and simulate.
This finds the corresponding oscillatory attractor due to the numerical perturbations.
We could converge to this attractor faster by adding a perturbation to `fp.u`.

````@example TLN
prob = ODEProblem(tln, fp.u, (0,150))
soln = solve(prob, Tsit5())
plot(soln)
````

Because root finders are only guaranteed to find local minima of the residual,
we start at a different initial guess and find a different attractor.

````@example TLN
prob = NonlinearProblem(tln, [1/3, 0, 1/2, 1/2, 1/2, 1/2])
fp = solve(prob)
````

We can plug in the fixed point and find the oscillatory attractor.

````@example TLN
prob = ODEProblem(tln, fp.u, (0,150))
soln = solve(prob, Tsit5())
plot(soln)
````

## Induced Subgraphs Preserve Attractors

When you take an induced subgraph. You can restrict the dynamics onto that subgraph.

````@example TLN
g = induced_subgraph(bt, [1,2,3])
tln = TLNetwork(CTLNetwork(g))
prob = NonlinearProblem(tln, [1/3, 0, 1/2])
fp = solve(prob)
support(fp.u)
prob = ODEProblem(tln, fp.u, (0,150))
soln = solve(prob, Tsit5())
plot(soln)
````

The following two functions automate the analysis that we did above
1. Restrict to a subgraph
2. Solve for a fixed point in the subgraph
3. Plug that solution in to the dynamics of the full system
4. Solve those dynamics and plot

```@docs; canonical=false
restriction_fixed_point
```

````@example TLN
function restriction_simulation(G, V, tspan=(0,150.0), parameters=DEFAULT_PARAMETERS)
  tln = CTLNetwork(G, parameters)
  σ, u₀ = restriction_fixed_point(G, V)
  @show u₀
  @show σ
  prob = ODEProblem(tln, u₀, tspan)
  soln = solve(prob, Tsit5())
  plt = plot(soln)
  return σ, soln, plt
end
````

## Mining patterns in our product graph
Let's take a look at our graph again

````@example TLN
draw(bt)
````

We can try finding an attractor from the triangle 2,4,6

````@example TLN
σ₀, soln, plt = restriction_simulation(bt, [2,4,6]);
plt
````

Here we can look at a tiny subgraph.

````@example TLN
σ₀, soln, plt = restriction_simulation(bt, [1,2]);
plt
````

There is a bigger support on 2,4,5,6

````@example TLN
σ₀, soln, plt = restriction_simulation(bt, [2,4,5,6]);
plt
````

Trying 1,2,4

````@example TLN
σ₀, soln, plt = restriction_simulation(bt, [1, 2,4]);
plt
````

Trying 1,2,3,5

````@example TLN
σ₀, soln, plt = restriction_simulation(bt, [1,2,3,5]);
plt
````

## Library Reference

You can access these functions from the module `AlgebraicDynamics.ThresholdLinear`.

```@autodocs
Modules = [AlgebraicDynamics.ThresholdLinear]
```