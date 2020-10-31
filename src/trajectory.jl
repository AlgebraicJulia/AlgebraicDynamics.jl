module Trajectory

using Catlab
using Catlab.CategoricalAlgebra
using ..DiscDynam

export trajectory

"""     trajectory(ds::AbstractDynamUWD, N, args...)

Computes the trajectory of a discrete dynamical system for N steps
"""

function trajectory(ds::AbstractDynamUWD, N, args...)
    k = nparts(ds, :State)

    xs = Array{Float64}(undef, N + 1, k)
    subpart(ds, :value)
    xs[1, : ] = subpart(ds, :value)

    for i in 1:N
        update!(view(xs,i+1, 1:k), ds, view(xs, i, 1:k), args...)
    end
    return xs
end

end #module