# AlgebraicDynamics

AlgebraicDynamics is a library for compositional dynamical systems. We build on
Catlab.jl to provide a software interfaces for the specification and solution of
dynamical systems with hierarchical structure.

## Examples

### Lotka Volterra Equations

```julia
using AlgebraicDynamics
using AlgebraicDynamics.DiscDynam

using Catlab
using Catlab.WiringDiagrams
using Catlab.CategoricalAlgebra
using Catlab.CategoricalAlgebra.CSets
using Catlab.Programs.RelationalPrograms

import AlgebraicDynamics.DiscDynam.functor

using Plots

α = 1.2
β = 0.1
γ = 1.3
δ = 0.1

h = 0.1
gen = Dict(
    :birth     => u -> u .+ h .* [ α*u[1]],
    :death     => u -> u .+ h .* [-γ*u[1]],
    :predation => u -> u .+ h.* [-β*u[1]*u[2], δ*u[1]*u[2]],
)

d = @relation (x,y) where (x, y) begin
    birth(x)
    predation(x,y)
    death(y)
end

lv = functor( Dict(:birth => Dynam(gen[:birth], 1, [1], [0]),
                    :death => Dynam(gen[:death], 1, [1], [0]),
                    :predation => Dynam(gen[:predation], 2, [1,2], [0,0])))(d)

n = 100
u = zeros(Float64, 4, n)
u[:,1] = [17.0, 17.0, 11.0, 11.0]
for i in 1:n-1
    @views update!(u[:, i+1], lv, u[:, i])
    println(u[:, i+1])
end

plot(u')
```

![Lotka Volterra Solutions](/docs/img/lvsoln.png)


This example is quite small and you could have worked it out by hand, but
this approach is based on solid mathematical theory that allows you to automatically
scale to larger and more complex problems.

## Theory

A dynamical systems is composed of subsystems that are wired together via shared variables. 
The connectivity structure is encoded into an undirected wiring diagram, which is defined in Catlab.
A simplified version is shown below

```julia
@present TheoryUWD(FreeSchema) begin
  Box::Ob
  Port::Ob
  Junction::Ob

  box::Hom(Port,Box)
  junction::Hom(Port,Junction)
end
```

The boxes represent primitive systems that expose interfaces defined by ports, so each port belongs to a box.
The junctions are places where systems can interact, each port maps to a junction, which is visually illustrated
as a wire connected the port's box to the junction node.

We extend this theory by adding States, and Dynamics to the wiring diagram. Each state belongs to a system (`Box`) and each
port points to a state, to indicate that the state is exposed via the port. Additionally each system has dynamics, which will be 
stored as a julia function that computes the action of that system on the state space.

```julia
@present TheoryDynamUWD <: TheoryUWD begin
    State::Ob
    Dynamics::Data

    system::Hom(State, Box)
    state::Hom(Port, State)
    dynamics::Attr(Box, Dynamics)
end
```

The vector field of a dynamical system can be computed compositionally from its definition as an undirected wiring diagram
the key idea is that resource sharing systems compose by superposition. Each system acts on its variables, and then the changes 
are summed around junctions that encode an equivalence relation on the variables.

```julia
function update!(newstate::AbstractVector, d::AbstractDynamUWD, state::AbstractVector, params...)
    # Apply the dynamics of each box on its incident states
    boxes = 1:nparts(d, :Box)
    for b in boxes
        states = incident(d, b, :system)
        dynamics = subpart(d, b, :dynamics)
        newvalues = update!(view(newstate, states), dynamics, view(state, states), params...)
    end

    # Apply the cumulative differences to appropriate junctions
    juncs = 1:nparts(d, :Junction)
    for j in juncs
        p = incident(d, j, :junction)
        length(p) > 0 || continue
        statesp = subpart(d, p, :state)
        nextval = state[first(statesp)] + mapreduce(i->newstate[i]-state[i], +, statesp, init=0)
        newstate[statesp] .= nextval
    end
    return newstate
end
```

## Future Work

Adding more integrators beyond the simple Euler's method. We need to include higher order polynomial methods and simplectic and implicit methods for physical problems. 
