# AlgebraicDynamics

AlgebraicDynamics is a library for compositional dynamical systems. We build on
Catlab.jl to provide a software interfaces for the specification and solution of
dynamical systems with hierarchical structure.

## Examples

You can specify and solve an SIR epidemic system using the following code:

```julia
using AlgebraicDynamics
using Plots
using Catlab
using Catlab.Doctrines
using RecursiveArrayTools
using OrdinaryDiffEq

# Declare that real numbers are modeled as Float64
R = Ob(FreeSMC, Float64)
# specify the S->I subsystem
si = Hom(System([1], [99.0, 1], [2], (x,t)->[-0.0035x[1]*x[2], 0.0035x[1]*x[2]]), R,R)
# specify the I->R subsystem
ir = Hom(System([1], [1.00, 0], [2], (x,t)->[-0.05x[1], 0.05x[1]]), R, R)
# compose them into SIR
sir = compose(si, ir)
# create an ODEProblem
p = problem(sir, (0,270.0))
# solve that ODEProblem using the regular solvers
sol = solve(p, alg=Tsit5())
# plot the solution, notice the ∀t u₂(t) == u₃(t) because they both represent I.
plot(sol)
```

We can then model what would happen in two noninteracting populations where one
population "flattened the curve" and the other did not.

```julia
# The first population uses the same parameters as before
si = Hom(System([1], [99.0, 1], [2], (x,t)->[-0.0035x[1]*x[2], 0.0035x[1]*x[2]]), R,R)
ir = Hom(System([1], [1.00, 0], [2], (x,t)->[-0.05x[1], 0.05x[1]]), R, R)
sir = compose(si, ir)
# The second population flattens the curve by reducing R₀
si_2 = Hom(System([1], [99.0, 1], [2], (x,t)->[-0.0025x[1]*x[2], 0.0025x[1]*x[2]]), R,R)
ir_2= Hom(System([1], [1.00, 0], [2], (x,t)->[-0.07x[1], 0.07x[1]]), R, R)
sir_2 = compose(si_2, ir_2)
# sir⊗sir_2 is the independent product system
p = problem(otimes(sir, sir_2), (0,270.0))
sol = solve(p, alg=Tsit5())
# notice that you have two sets of SIR curves lagging each other
plot(sol)
```

These examples are quite small and you could have worked them out by hand, but
this approach is based on solid mathematical theory that allows you to automatically
scale to larger and more complex problems.

## Theory

Each system is represented by a term in a Generalized Algebraic Theory (GAT).
Within a GAT, there are operators that allow you to combine terms into bigger terms,
for example we have used `f⋅g` for composition and `f⊗g` for independent combination.
Because this system is based on the algebraic perspective implemented in Catlab,
we can build arbitrarily complex systems by building formulas. The behavior of
the combined system is defined by a structure preserving map into vector fields.

Here is a simplified version of the definition of combination and composition:
```julia
function forward(product::FreeSMC.Hom{:otimes}, u, t)
    # to compute the tangent vector of `f⊗g` compute the tangent vector of f and g
    f,g = product.args[1], product.args[2]
    duf = forward(f, u.x[1], t)
    dug = forward(g, u.x[2], t)
    # and then concatenate them
    du = ArrayPartition(duf, dug)
    return du
end

function forward(composite::FreeSMC.Hom{:compose}, u, t)
    # to compute the tangent vector of `f⋅g` compute the tangent vector of f and g
    f,g = composite.args[1], composite.args[2]
    duf = forward(f, u.x[1], t)
    dug = forward(g, u.x[2], t)
    # then ensure that they are equal on the variables they share
    # we use symmetric superposition to ensure the following equality
    # forward(f,u,t)[codom(f)] == forward(g,u,t)[dom(g)]
    dufup = duf[codom_ids(f)]
    dugup = dug[dom_ids(g)]
    duf[codom_ids(f)] .+= dugup
    dug[  dom_ids(g)] .+= dufup
    # and then concatenate the tangent vectors
    du = ArrayPartition(duf, dug)
    return du
end
```
