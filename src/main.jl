using Catlab
using Catlab.Doctrines
using Catlab.WiringDiagrams
using Catlab.Programs
using Catlab.Graphics
import Convex, SCS

R = Ob(FreeSymmetricMonoidalCategory, Float64)
sine = Hom(:sin, R, R)

draw(x) = to_composejl(x)#, orientation=LeftToRight)

draw(sine)
module Systems
using Catlab
using Catlab.Doctrines
using Catlab.WiringDiagrams
using Catlab.Programs
using Catlab.Graphics
import Convex, SCS

const FreeSMC = FreeSymmetricMonoidalCategory
struct System{T, F}
    inputs::Vector{Int}
    states::Vector{T}
    outputs::Vector{Int}
    f::F
end

sinesys = System(Int[], [1.0], [1], x -> -x)

forward(sys::System, t) = begin
    du = sys.f(sys.states, t)
end


function forward(f::FreeSMC.Hom{:generator}, t)
    sys = f.args[1]
    return sys.f(sys.states, t)
end

function forward(f::FreeSMC.Hom{:id}, t)
    space = f.args[1]
    return zeros(Float64, ndims(space))
end


dom_ids(f::FreeSMC.Hom{:generator}) = f.args[1].inputs
codom_ids(f::FreeSMC.Hom{:generator}) = f.args[1].outputs
nstates(f::FreeSMC.Hom{:generator}) = length(f.args[1].states)
nstates(f::FreeSMC.Hom{:otimes}) = sum(nstates(sys) for sys in f.args)
nstates(f::FreeSMC.Hom{:compose}) = sum(nstates(sys) for sys in f.args)

dom_ids(f::FreeSMC.Hom{:otimes}) = vcat(
  dom_ids(f.args[1]),
  dom_ids(f.args[2]) .+ nstates(f.args[1])
)
codom_ids(f::FreeSMC.Hom{:otimes}) = vcat(
  codom_ids(f.args[1]),
  codom_ids(f.args[2]) .+ nstates(f.args[1])
)
dom_ids(f::FreeSMC.Hom{:compose}) = dom_ids(f.args[1])
codom_ids(f::FreeSMC.Hom{:compose}) = codom_ids(f.args[2]) .+ nstates(f.args[1])

function forward(composite::FreeSMC.Hom{:compose}, t)
    f,g = composite.args[1], composite.args[2]
    @show duf = forward(f, t)
    @show dug = forward(g, t)

    dufup = duf[codom_ids(f)]
    dugup = dug[dom_ids(g)]
    duf[codom_ids(f)] .+= dugup
    dug[  dom_ids(g)] .+= dufup
    @show du = vcat(duf, dug)
    return du
end
function forward(product::FreeSMC.Hom{:otimes}, t)
    f,g = product.args[1], product.args[2]
    duf = forward(f, t)
    dug = forward(g, t)
    du = vcat(duf, dug)
    return du
end

using Test
R = Ob(FreeSMC, Float64)
id(R)
@test forward(id(R), 1.0) == zeros(Float64, 1)
@test forward(id(otimes(R, R)), 1.0) == zeros(Float64, 2)
@test forward(id(otimes(R, otimes(R, R))), 1.0) == zeros(Float64, 3)

f = Hom(System([1], [1.0], [1], (x, t)->-x), R, R)
@test forward(f, 1) == [-1]
@test forward(compose(f,f), 1) == [-2, -2]

@test dom_ids(compose(f,f)) == [1]
@test codom_ids(compose(f,f)) == [2]

@test dom_ids(otimes(f,f)) == [1, 2]
@test codom_ids(otimes(f,f)) == [1, 2]

@test dom_ids(otimes(compose(f,f),f)) == [1, 3]
@test codom_ids(otimes(compose(f,f),f)) == [2, 3]

@test forward(otimes(f,f), 1) == [-1,-1]
@test forward(otimes(compose(f,f),f), 1) == [-2, -2,-1]

si = Hom(System([1], [99, 1], [2], (x,t)->[-0.0035x[1]*x[2], 0.0035x[1]*x[2]]), R,R)
ir = Hom(System([1], [1, 0], [2], (x,t)->[-0.05x[1], 0.05x[1]]), R, R)
sir = compose(si, ir)
sirtest = Hom(System([1], [99,1,0], [4], (x,t)->[-0.0035x[1]*x[2], 0.0035x[1]*x[2]+ -0.05x[2], 0.05x[2]]), R, R)

@test forward(sir, 1)[[1,2,end]] == forward(sirtest, 1)

using RecursiveArrayTools
# import RecursiveArrayTools: VectorOfArray
v = ArrayPartition([1,2,3],[4,5,6])
@test v[1] == 1
@test v[2] == 2
@test v[5] == 5
@test v.x[1] == [1,2,3]
@test v.x[2] == [4,5,6]

v = ArrayPartition([1,2,3], ArrayPartition([4,5],[6]), [7,8,9])
@test v[5] == 5
@test v[6] == 6

function states(f::FreeSMC.Hom{:generator})
    return f.args[1].states
end

function states(composite::FreeSMC.Hom{:compose})
    return ArrayPartition((map(states, composite.args))...)
end

function states(product::FreeSMC.Hom{:otimes})
    return ArrayPartition((map(states, product.args))...)
end

@show states(si)
@show states(ir)
@show states(sir)
@test states(sir)[1] == 99
@test states(sir)[2] == 1
@test states(sir)[3] == 1
end
