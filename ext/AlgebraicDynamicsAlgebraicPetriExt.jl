module AlgebraicDynamicsAlgebraicPetriExt

using AlgebraicPetri
using AlgebraicDynamics
using Catlab
using ComponentArrays
import AlgebraicDynamics.UWDDynam: ContinuousResourceSharer

export OpenNet

"""    OpenNet=Union{OpenPetriNet,OpenLabelledPetriNet,OpenLabelledReactionNet}

A type alias for any kind of open network from AlgebraicPetri.
"""
const OpenNet = Union{OpenPetriNet,OpenLabelledPetriNet,OpenLabelledReactionNet}

# vecfields need to be initialized differently depending on type of net
function dynamics(pn::OpenPetriNet,T::Type,ns::Int64)
  f! = vectorfield(apex(pn))
  storage = zeros(T,ns)
  vf(u, p, t) = begin f!(storage, u, p, t); return storage end
end


function dynamics(pn::OpenLabelledPetriNet,T::Type,ns::Int64)
  f! = vectorfield(apex(pn))
  storage = ComponentArray(NamedTuple{tuple(snames(apex(pn))...)}(zeros(T,ns)))
  vf(u, p, t) = begin f!(storage, u, p, t); return storage end
  return vf
end

function dynamics(pn::OpenLabelledReactionNet,T::Type,ns::Int64)
  f! = vectorfield(apex(pn))
  storage = ComponentArray(NamedTuple{tuple(snames(apex(pn))...)}(zeros(T,ns)))
  rt = rates(apex(pn))
  vf(u, p, t) = begin f!(storage, u, rt, t); return storage end
  return vf
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