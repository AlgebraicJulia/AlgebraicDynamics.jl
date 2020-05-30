module Systems
using Catlab
using Catlab.Doctrines
using Catlab.WiringDiagrams
using Catlab.Programs
using RecursiveArrayTools
using OrdinaryDiffEq

export FreeSMC, System, dom_ids, codom_ids, nstates, states,
 forward, initforward, problem

const FreeSMC = FreeSymmetricMonoidalCategory

struct System{T, F}
    inputs::Vector{Int}
    states::Vector{T}
    outputs::Vector{Int}
    f::F
end

dom_ids(f::FreeSMC.Hom{:generator}) = f.args[1].inputs
codom_ids(f::FreeSMC.Hom{:generator}) = f.args[1].outputs

nstates(f::FreeSMC.Hom{:generator}) = length(f.args[1].states)
nstates(f::FreeSMC.Hom{:otimes}) = sum(nstates(sys) for sys in f.args)
nstates(f::FreeSMC.Hom{:compose}) = sum(nstates(sys) for sys in f.args)

dom_ids(f::FreeSMC.Hom{:otimes}) = ArrayPartition(
  dom_ids(f.args[1]),
  dom_ids(f.args[2]) .+ nstates(f.args[1])
)
codom_ids(f::FreeSMC.Hom{:otimes}) = ArrayPartition(
  codom_ids(f.args[1]),
  codom_ids(f.args[2]) .+ nstates(f.args[1])
)
dom_ids(f::FreeSMC.Hom{:compose}) = dom_ids(f.args[1])
codom_ids(f::FreeSMC.Hom{:compose}) = codom_ids(f.args[2]) .+ nstates(f.args[1])

function states(f::FreeSMC.Hom{:generator})
    return f.args[1].states
end

function states(f::FreeSMC.Hom{:id})
    return ArrayPartition(zeros(Float64, ndims(dom(f))))
end

function states(composite::FreeSMC.Hom{:compose})
    #@assert foldl(==, map(states, composite.args)) "initial conditions are incorrectly specified"
    return ArrayPartition((map(states, composite.args))...)
end

function states(product::FreeSMC.Hom{:otimes})
    return ArrayPartition((map(states, product.args))...)
end

function forward(sys::System, u, t)
    du = sys.f(u, t)
end

function forward(f::FreeSMC.Hom{:generator}, u, t)
    sys = f.args[1]
    return forward(sys, u, t)
end

function forward(f::FreeSMC.Hom{:id}, u, t)
    return zero(u)
end

function forward(composite::FreeSMC.Hom{:compose}, u, t)
    f,g = composite.args[1], composite.args[2]
    duf = forward(f, u.x[1], t)
    dug = forward(g, u.x[2], t)

    dufup = duf[codom_ids(f)]
    dugup = dug[dom_ids(g)]
    duf[codom_ids(f)] .+= dugup
    dug[  dom_ids(g)] .+= dufup
    du = ArrayPartition(duf, dug)
    return du
end
function forward(product::FreeSMC.Hom{:otimes}, u, t)
    f,g = product.args[1], product.args[2]
    duf = forward(f, u.x[1], t)
    dug = forward(g, u.x[2], t)
    du = ArrayPartition(duf, dug)
    return du
end

initforward(f, t) = forward(f, states(f), t)

problem(f::FreeSMC.Hom{T}, tspan) where {T} = ODEProblem((u,p,t)->forward(f,u,t), states(f), tspan)

end
