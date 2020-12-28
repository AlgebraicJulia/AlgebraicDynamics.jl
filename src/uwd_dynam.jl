using Catlab
using Catlab.WiringDiagrams
using Catlab.CategoricalAlgebra
using Catlab.CategoricalAlgebra.FinSets
using Catlab.Theories


using Catlab.WiringDiagrams.UndirectedWiringDiagrams: AbstractUWD
import Catlab.WiringDiagrams: oapply


const UWD = UndirectedWiringDiagram

struct ResourceSharer{T}
  nports::Int
  nstates::Int
  dynamics::Function
  portmap::Vector{Int64}
end

nstates(r::ResourceSharer) = r.nstates
nports(r::ResourceSharer)  = r.nports
portmap(r::ResourceSharer) = FinFunction(r.portmap, nstates(r))
eval_dynamics(r::ResourceSharer, u, args...) = r.dynamics(u, args...)
eval_dynamics!(du, r::ResourceSharer, u, args...) = begin
    du .= eval_dynamics(r, u, args...)
end

show(io::IO, vf::ResourceSharer) = print("ResourceSharer(ℝ^$(vf.nstates) → ℝ^$(vf.nstates)) with $(vf.nports) exposed ports")


function fills(r::ResourceSharer, d::AbstractUWD, b::Int)
  b <= nparts(d, :Box) || error("Trying to fill box $b, when $d has fewer that $b boxes")
  return nports(r) == length(incident(d, b, :box))
end

function oapply(d::AbstractUWD, xs::Vector{ResourceSharer{T}}) where T
  for box in parts(d, :Box)
      fills(xs[box], d, box) || error("$xs[box] does not fill box $box")
  end
  
  S = coproduct((FinSet∘nstates).(xs))
  states(b::Int) = legs(S)[b].func

  P = coproduct((FinSet∘nports).(xs))
  port_function = copair([compose( portmap(xs[i]), legs(S)[i]) for i in 1:length(xs)])
  
  S′ = pushout(port_function, FinFunction(subpart(d, :junction), nparts(d, :Junction)))
  state_map = legs(S′)[1]
  
  
  function v(u′,args...)
      u = getindex(u′,  state_map.func)
      du = zero(u)
      # apply dynamics
      for b in parts(d, :Box)
        eval_dynamics!(view(du, states(b)), xs[b], view(u, states(b)), 0, 0)
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



d = UWD(2)
add_parts!(d, :Box, 3)
add_parts!(d, :Junction, 2, outer_junction = [1,2])
add_parts!(d, :Port, 4, box=[1,2,2,3], junction=[1,1,2,2])

α, β, γ, δ = 0.3, 0.015, 0.015, 0.7

dotr(x,p,t)  = α*x
dotrf(x,p,t) = [-β*x[1]*x[2], γ*x[1]*x[2]]
dotf(x,p,t)  = -δ*x

rabbit_growth       = ResourceSharer{Float64}(1, 1, dotr,  [1])
rabbitfox_predation = ResourceSharer{Float64}(2, 2, dotrf, [1,2])
fox_decline         = ResourceSharer{Float64}(1, 1, dotf,  [1])

xs = [rabbit_growth, rabbitfox_predation, fox_decline]

rf = oapply(d, xs)