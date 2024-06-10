````@meta
Draft=false
````
# AlgebraicPetri.jl Integration

You can construct a [`ContinuousResourceSharer`](@ref) from an Open Petri Net for any kind of network supported by [AlgebraicPetri.jl](https://algebraicjulia.github.io/AlgebraicPetri.jl/dev/) including:

1. OpenPetriNet
2. OpenLabelledPetriNet
3. OpenLabelledReactionNet

````@example AlgPetri
using AlgebraicPetri
using AlgebraicDynamics
using Catlab.Graphics
Brusselator = LabelledPetriNet([:A, :B, :D, :E, :X, :Y],
  :t1 => (:A => (:X, :A)),
  :t2 => ((:X, :X, :Y) => (:X, :X, :X)),
  :t3 => ((:B, :X) => (:Y, :D, :B)),
  :t4 => (:X => :E)
)
to_graphviz(Brusselator)
````

You just call the constructor for ContinuousResourceSharer on an Open Petri Net. 
The constructor knows where to use labels or integers for the state variables.
It also knows to get parameters from the ReactionNet and enclose them into the dynamics.

````@example AlgPetri
open_bruss = Open([:A, :D], Brusselator, [:A, :B])
rs = ContinuousResourceSharer{Float64}(open_bruss)
````