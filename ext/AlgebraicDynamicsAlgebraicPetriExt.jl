module AlgebraicDynamicsAlgebraicPetriExt

using AlgebraicPetri
using AlgebraicDynamics
using Catlab
import AlgebraicDynamics.UWDDynam: ContinuousResourceSharer

function ContinuousResourceSharer{T}(pn::Union{OpenPetriNet, OpenLabelledPetriNet}) where T
    nstates = nparts(apex(pn), :S)
    portmap = vcat(map(legs(pn)) do f
      f[:S](parts(dom(f), :S))
    end...)
    nports = length(portmap)
    vf(u, p, t) = vectorfield(apex(pn))(zeros(nstates), u, p, t)
  
    ContinuousResourceSharer{T}(nports, nstates, vf, portmap)
  end

end