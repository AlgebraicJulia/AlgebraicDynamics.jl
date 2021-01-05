module AlgebraicDynamics

using Catlab
using Catlab.Theories
using Catlab.WiringDiagrams
using Catlab.Programs

include("uwd_dynam.jl")
include("dwd_dynam.jl")

include("linrels.jl")
include("systems.jl")
include("hypergraphs.jl")
include("discdynam.jl")
end # module
