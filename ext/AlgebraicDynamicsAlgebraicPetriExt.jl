module AlgebraicDynamicsAlgebraicPetriExt

using AlgebraicPetri
using AlgebraicDynamics
using Catlab
using LabelledArrays
import AlgebraicDynamics.UWDDynam: ContinuousResourceSharer

export OpenNet

"""    OpenNet=Union{OpenPetriNet,OpenLabelledPetriNet,OpenLabelledReactionNet}

A type alias for any kind of open network from AlgebraicPetri.
"""
const OpenNet = Union{OpenPetriNet,OpenLabelledPetriNet,OpenLabelledReactionNet}

# vecfields need to be initialized differently depending on type of net
function dynamics(pn::OpenPetriNet,T::Type,ns::Int64)
  vf(u, p, t) = vectorfield(apex(pn))(zeros(T,ns), u, p, t)
end

function dynamics(pn::OpenLabelledPetriNet,T::Type,ns::Int64)
  vf(u, p, t) = vectorfield(apex(pn))(LVector(NamedTuple{tuple(snames(apex(pn))...)}(zeros(T,ns))), u, p, t)
end

function dynamics(pn::OpenLabelledReactionNet,T::Type,ns::Int64)
  vf(u, p, t) = vectorfield(apex(pn))(LVector(NamedTuple{tuple(snames(apex(pn))...)}(zeros(T,ns))), u, rates(apex(pn)), t)
end

function ContinuousResourceSharer{T}(pn::OpenNet) where T
  nstates = nparts(apex(pn), :S)
  portmap = vcat(map(legs(pn)) do f
    f[:S](parts(dom(f), :S))
  end...)
  nports = length(portmap)
  ContinuousResourceSharer{T}(nports, nstates, dynamics(pn,T,nstates), portmap)
end
end