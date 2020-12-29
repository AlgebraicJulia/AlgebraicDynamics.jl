module UWDDynam
using Catlab
using Catlab.WiringDiagrams
using Catlab.CategoricalAlgebra
using Catlab.CategoricalAlgebra.FinSets
using Catlab.Theories


using Catlab.WiringDiagrams.UndirectedWiringDiagrams: AbstractUWD
import Catlab.WiringDiagrams: oapply

export ResourceSharer, nstates, nports, portmap, portfunction, 
eval_dynamics, eval_dynamics!, exposed_states, fills, induced_states

import Base: show

const UWD = UndirectedWiringDiagram

struct ResourceSharer{T}
  nports::Int
  nstates::Int
  dynamics::Function
  portmap::Vector{Int64}
end

ResourceSharer{T}(nstates::Int, dynamics::Function) where T = 
    ResourceSharer{T}(nstates,nstates, dynamics, Vector{Int64}(1:nstates))

nstates(r::ResourceSharer) = r.nstates
nports(r::ResourceSharer)  = r.nports
portmap(r::ResourceSharer) = r.portmap
portfunction(r::ResourceSharer) = FinFunction(r.portmap, nstates(r))
eval_dynamics(r::ResourceSharer, u, args...) = r.dynamics(u, args...)
eval_dynamics!(du, r::ResourceSharer, u, args...) = begin
    du .= eval_dynamics(r, u, args...)
end
exposed_states(r::ResourceSharer, u) = getindex(u, portmap(r))

show(io::IO, vf::ResourceSharer) = print("ResourceSharer(ℝ^$(vf.nstates) → ℝ^$(vf.nstates)) with $(vf.nports) exposed ports")


function fills(r::ResourceSharer, d::AbstractUWD, b::Int)
  b <= nparts(d, :Box) || error("Trying to fill box $b, when $d has fewer that $b boxes")
  return nports(r) == length(incident(d, b, :box))
end

function induced_states(d::AbstractUWD, xs::Vector{ResourceSharer{T}}) where T
    for box in parts(d, :Box)
        fills(xs[box], d, box) || error("$xs[box] does not fill box $box")
    end
    
    S = coproduct((FinSet∘nstates).(xs))  
    P = coproduct((FinSet∘nports).(xs))
    total_portfunction = copair([compose( portfunction(xs[i]), legs(S)[i]) for i in 1:length(xs)])
    
    return pushout(total_portfunction, FinFunction(subpart(d, :junction), nparts(d, :Junction)))
end


oapply(d::AbstractUWD, xs::Vector{ResourceSharer{T}}) where T = 
    oapply(d, xs, induced_states(d, xs))

function oapply(d::AbstractUWD, xs::Vector{ResourceSharer{T}}, S′::Pushout) where T
  state_map = legs(S′)[1]
  S = coproduct((FinSet∘nstates).(xs))
  states(b::Int) = legs(S)[b].func

  function v(u′, args...)
      u = getindex(u′,  state_map.func)
      du = zero(u)
      # apply dynamics
      for b in parts(d, :Box)
        eval_dynamics!(view(du, states(b)), xs[b], view(u, states(b)), args...)
      end
      # add along junctions
      du′ = [sum(Array{T}(view(du, preimage(state_map, i)))) for i in codom(state_map)]
      return du′
  end
  
  junction_map = legs(S′)[2]
  outer_junction_map = FinFunction(subpart(d, :outer_junction), nparts(d, :Junction))

  return ResourceSharer{T}(
    nparts(d, :OuterPort), 
    length(apex(S′)), 
    v, 
    compose(outer_junction_map, junction_map).func)
end

end #module