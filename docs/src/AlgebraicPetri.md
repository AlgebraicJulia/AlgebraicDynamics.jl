# AlgebraicPetri.jl Integration

You can construct a [`ContinuousResourceSharer`](@ref) from an Open Petri Net for any kind of network supported by [AlgebraicPetri.jl](https://algebraicjulia.github.io/AlgebraicPetri.jl/dev/) including:

1. OpenPetriNet
2. OpenLabelledPetriNet
3. OpenLabelledReactionNet

````@example AlgPetri
using AlgebraicPetri
using AlgebraicDynamics
using Catlab.Graphics
Brusselator = PetriNet(6,
  (1 => (5, 1)),
  ((5, 5, 6) => (5, 5, 5)),
  ((2, 5) => (6, 3, 2)),
  (5 => 4)
)
to_graphviz(Brusselator)
````
You just call the constructor for ContinuousResourceSharer on an Open Petri Net. 
The constructor knows where to use labels or integers for the state variables.
It also knows to get parameters from the ReactionNet and enclose them into the dynamics.


````@example AlgPetri
open_bruss = Open([1, 3], Brusselator, [1, 2])
rs = ContinuousResourceSharer{Float64}(open_bruss)
````

For the PetriNet case, you supply the state and parameters with regular arrays.

````@example AlgPetri
using OrdinaryDiffEq
using Plots
tspan = (0.0,100.0)
params = [1.0, 1., 1., 1.0]
u0 = [1,3.7,0.0,0.0,1,1]
prob = ODEProblem(rs, u0, tspan, params)
soln = solve(prob, Tsit5())
plot(soln, idxs=[5,6])
````

For the LabelledPetriNet case, you supply the state and parameters with `LArray`.

````@example AlgPetri
Brusselator = LabelledPetriNet([:A, :B, :D, :E, :X, :Y],
  :t1 => (:A => (:X, :A)),
  :t2 => ((:X, :X, :Y) => (:X, :X, :X)),
  :t3 => ((:B, :X) => (:Y, :D, :B)),
  :t4 => (:X => :E)
)
to_graphviz(Brusselator)
````

````@example AlgPetri
open_bruss = Open([:A, :D], Brusselator, [:A, :B])
rs = ContinuousResourceSharer{Float64}(open_bruss)
````

    
````julia
using ComponentArrays
tspan = (0.0,100.0)
params = ComponentArray(t1=1.0, t2=1.2, t3=3.14, t4=0.1)
u0 = ComponentArray(A=1.0, B=3.17, D=0.0, E=0.0, X=1.0, Y=1.9)
eval_dynamics(rs, u0, params, 0.0)
prob = ODEProblem((u,p,t) -> eval_dynamics(rs, u, p, t), u0, tspan, params)
sol = solve(prob, Tsit5())
plot(sol, idxs=[:X, :Y])
````