module AlgebraicDynamics

using Reexport

include("uwd_dynam.jl")
include("dwd_dynam.jl")
include("cpg_dynam.jl")

@reexport using .DWDDynam
@reexport using .UWDDynam
@reexport using .CPortGraphDynam

end # module
