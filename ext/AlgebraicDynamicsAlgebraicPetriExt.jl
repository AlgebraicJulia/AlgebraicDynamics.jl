module AlgebraicDynamicsAlgebraicPetriExt

using AlgebraicPetri
using AlgebraicDynamics
using Catlab
using LabelledArrays
import AlgebraicDynamics.UWDDynam: ContinuousResourceSharer

# vecfields need to be initialized differently depending on type of net
function vf(pn::OpenPetriNet,T::Type,ns::Int64)
  vf(u, p, t) = vectorfield(apex(pn))(zeros(T,ns), u, p, t)
end

function vf(pn::OpenLabelledPetriNet,T::Type,ns::Int64)
  vf(u, p, t) = vectorfield(apex(pn))(LVector(NamedTuple{tuple(snames(apex(pn))...)}(zeros(T,ns))), u, p, t)
end

function vf(pn::OpenLabelledReactionNet,T::Type,ns::Int64)
  vf(u, p, t) = vectorfield(apex(pn))(LVector(NamedTuple{tuple(snames(apex(pn))...)}(zeros(T,ns))), u, rates(apex(pn)), t)
end

function ContinuousResourceSharer{T}(pn::Union{OpenPetriNet,OpenLabelledPetriNet,OpenLabelledReactionNet}) where T
  nstates = nparts(apex(pn), :S)
  portmap = vcat(map(legs(pn)) do f
    f[:S](parts(dom(f), :S))
  end...)
  nports = length(portmap)
  ContinuousResourceSharer{T}(nports, nstates, vf(pn,T,nstates), portmap)
end
end