# AlgebraicPetri.jl Integration

You can construct a [`ContinuousResourceSharer`](@ref) from an Open Petri Net for any kind of network supported by AlgebraicPetri.jl including:

1. OpenPetriNet
2. OpenLabelledPetriNet
3. OpenLabelledReactionNet

````@example
Brusselator = LabelledPetriNet([:A, :B, :D, :E, :X, :Y],
  :t1 => (:A => (:X, :A)),
  :t2 => ((:X, :X, :Y) => (:X, :X, :X)),
  :t3 => ((:B, :X) => (:Y, :D, :B)),
  :t4 => (:X => :E)
)

open_bruss = Open([:A, :D], Brusselator, [:A, :B])
rs = ContinuousResourceSharer{Float64}(open_bruss)
````