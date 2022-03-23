module AlgebraicDynamics

using Catlab
using Catlab.Theories
using Catlab.WiringDiagrams
using Catlab.Programs

include("uwd_dynam.jl")
include("dwd_dynam.jl")
include("cpg_dynam.jl")
include("trajectories.jl")
include("dwd_dils.jl")

end # module
