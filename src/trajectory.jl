module Trajectory

using Catlab
using Catlab.CategoricalAlgebra
using ..DiscDynam

export trajectory, eulers


function eulers(dotf::Function)
    return (x0, h) -> x0 + h*dotf(x0)
end 

function eulers(dotf::Function, max_step::Number)
    function step(x,h)
        for _ in 1:(h/max_step)
            x = x + max_step*dotf(x)
        end
        x = x + (h%max_step)*dotf(x)
        return x
    end
end 

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