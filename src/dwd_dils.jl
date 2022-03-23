module DWDDILS
export Monomial, tensor, positions, directions, DILS, input_polys, input_poly,
       output_polys, output_poly, nstates, forward_readout, backward_readout,
       eval_dynamics, isa_state, sequential_compose, layerize,
       generalized_kroenecker, oapply, unit


using Test
using LinearAlgebra

using Catlab.WiringDiagrams.DirectedWiringDiagrams
using Catlab.Graphs, Catlab.CategoricalAlgebra


"""

A monomial is a pair of integers  A => B representing the poly ``\\mathbb{R}^By^{\\mathbb{R}^A}``
"""

const Monomial = Pair{Int,Int}
tensor(ms::Vector{Monomial}) = sum(first.(ms)) => sum(last.(ms))

positions(m::Monomial) = m.second
directions(m::Monomial) = m.first
positions(ms::Vector{Monomial}) = sum(first.(ms))


struct DILS{T}
  input_polys::Vector{Monomial}
  output_polys::Vector{Monomial}
  nstates::Int # dimension of the state space
  forward_readout::Function
  backward_readout::Function
  eval_dynamics::Function
end

DILS{T}(input_poly::Monomial, output_poly::Monomial, nstates::Int,
        forward_readout::Function, backward_readout::Function,
        eval_dynamics::Function) where T =
  DILS{T}([input_poly], [output_poly], nstates, forward_readout,
          backward_readout, eval_dynamics)



input_polys(d::DILS) = d.input_polys
output_polys(d::DILS) = d.output_polys
input_poly(d::DILS) = tensor(input_polys(d))
output_poly(d::DILS) = tensor(output_polys(d))
input_poly(d::DILS, i::Int) = input_polys(d)[i]
output_poly(d::DILS, i::Int) = output_polys(d)[i]

nstates(d::DILS) = d.nstates

""" forward_readout(d::DILS, u::AbstractVector, x::AbstractVector)

This is half of the map on positions for the DILS `d` given by
  ``Sy^S \\to [p,q]``. It takes a state ``u \\in \\mathbb{R}^S``
and a position in the input polynomial ``x \\in p(1)`` and returns a position of
the output polynomial.
"""

forward_readout(d::DILS, u::AbstractVector, x::AbstractVector) =
  d.forward_readout(u, x)

""" backward_readout(d::DILS, u::AbstractVector, x::AbstractVector, y::AbstractVector)

This is the other half of the map on positions. It takes a state, a position on
the input polynomial and a position on the output polynomial and returns a
direction on the input polynomial.

``\\sum_{s \\in S} \\sum_{i\\in p(1)} q[f(s,i)] \\rightarrow p[i]``
"""

backward_readout(d::DILS, u::AbstractVector, x::AbstractVector, y::AbstractVector) =
  d.backward_readout(u, x, y)

""" eval_dynamics(d::DILS, u::AbstractVector, x::AbstractVector, y::AbstractVector)

This is the map on directions of the DILS `d` given by
``{\\mathbb{R}^S}y^{\\mathbb{R}^S} \\to [p,q]`. It takes a state
``u \\in \\mathbb{R}^S``, a position ``x \\in p(1)``, a direction
``y \\in q[f(u,x)]`` (where ``f`` is the `forward_readout`)
and returns an updated state in ``\\mathbb{R}^S``
"""

eval_dynamics(d::DILS, u::AbstractVector, x::AbstractVector,
              y::AbstractVector) = d.eval_dynamics(u, x, y)

isa_state(d::DILS{T}, u::AbstractVector{T}) where T = length(u) == nstates(d)


# A DILS based on a function
function DILS{T}(f::Function, ninput::Int, noutput::Int; monomial=true) where T
  DILS{T}(
    monomial ? 0=>ninput : [0=>1 for _ in 1:ninput],  # input poly is ninput y^0
    monomial ? 0=>noutput : [0=>1 for _ in 1:noutput], # out poly is noutput y^0
    0,          # trivial state
    (_, x) -> f(x),  #forward map
    (_, _, _) -> T[], # backward map
    (_, _, _) -> T[]  # eval dynamics
  )
end

# A DILS that outputs constant values
DILS{T}(c::AbstractVector; monomial=true) where T =
  DILS{T}(_ -> c, 0, length(c); monomial=monomial)

"""
Basically a (dependent) lens
"""
function DILS{T}(forward::Function, backward::Function, ninput::Int,
                 noutput::Int) where T
  DILS{T}(
    [ninput=>ninput],
    [noutput=>noutput],
    0,                     # trivial state
    (_, x) -> forward(x),  #forward map
    (_, x, y) -> backward(x,y), # backward map
    (_, _, _) -> T[]  # eval dynamics
    )
end


"""
Compose two DILS in sequences

TODO: Change braiding to be any wiring pattern from d1 to d2.
If the output poly and input poly are of the form Ay^A (i.e. same inputs and outputs) then
you can do this by constructing an integer-valued matrix based on the wiring pattern encoding copying, merging, delete, and create.
For the forward pass multiply the positions of the output of d1 by the matrix.
For the backward pass, multiply the directions of the input of d2 by the transpose of the matrix.
"""

function sequential_compose(d1::DILS{T}, d2::DILS{T},
    weights::Union{Nothing,AbstractMatrix}=nothing)::DILS{T} where T

  if isnothing(weights)
    op, ip = output_polys(d1), input_polys(d2)
    if op == ip
      weights = I(length(op))
    else
      error("Cannot infer sequential compose weights between $op and $ip")
    end
  end

  # check that the monomials match
  for (i,k) in pairs(weights)
    if k > 0
      input_poly(d2, i[1]) == output_poly(d1, i[2])
    end
  end
  d1_state = u -> u[1:d1.nstates]
  d2_state = u -> u[d1.nstates+1:end]
  d1_output = (u,x) -> forward_readout(d1, d1_state(u), x)
  d2_backward = (u,x,y) -> backward_readout(d2, d2_state(u), d1_output(u,x), y)

  # This produces the matrices that copy and add the things flowing along the wires correctly.
  forward_matrix = generalized_kroenecker(weights,
                                          map(positions, input_polys(d2)),
                                          map(positions, output_polys(d1)))
  backward_matrix = generalized_kroenecker(weights',
                                           map(directions, output_polys(d1)),
                                           map(directions, input_polys(d2)))
  return DILS{T}(
    input_polys(d1), # input monomials
    output_polys(d2), # output monomials
    nstates(d1)+nstates(d2), # number of states/dimensions
    (u,x) -> forward_readout(d2, d2_state(u), forward_matrix*d1_output(u,x)),
    (u,x,y) -> backward_readout(d1, d1_state(u), x, backward_matrix*d2_backward(u,x,y)),
    (u,x,y) -> vcat(eval_dynamics(d1, d1_state(u), x, backward_matrix'*d2_backward(u, x, y)), # updated state for d1
                    eval_dynamics(d2, d2_state(u), forward_matrix*d1_output(u,x), y))       # updated state for d2
  )
end

"""TODO TEST"""
function unit(::Type{DILS{T}}) where T
  function fun2(u, x)
    all(isempty.([u,x])) || error("Bad input")
    T[]
  end
  function fun3(u, x, y)
    all(isempty.([u,x,y])) || error("Bad input")
    T[]
  end

  DILS{T}(Monomial[],Monomial[], 0, fun2, fun3, fun3)
end

"""TODO TEST BACKWARD READOUT AND DYNAMICS"""
function tensor(d1::DILS{T}, d2::DILS{T}) where T
  d1_state = u -> u[1:d1.nstates]
  d2_state = u -> u[d1.nstates+1:end]
  d1_in = x -> x[1:positions(input_poly(d1))]
  d2_in = x -> x[positions(input_poly(d1))+1:end]
  d1_out = y-> y[1:directions(output_poly(d1))]
  d2_out = y-> y[directions(output_poly(d1))+1:end]
  DILS{T}(vcat(input_polys(d1), input_polys(d2)),
          vcat(output_polys(d1), output_polys(d2)),
          d1.nstates + d2.nstates,
          (u,x) -> vcat(forward_readout(d1, d1_state(u), d1_in(x)),
                        forward_readout(d2, d2_state(u), d2_in(x))),
          (u,x,y) -> vcat(backward_readout(d1, d1_state(u), d1_in(x), d1_out(y)),
                          backward_readout(d2, d2_state(u), d2_in(x), d2_out(y))),
          (u,x,y) -> vcat(eval_dynamics(d1, d1_state(u), d1_in(x),d1_out(y)),
                          eval_dynamics(d2, d2_state(u), d2_in(x),d2_out(y)))
  )
end

"""TODO TEST"""
function tensor(ds::AbstractVector{<:DILS{T}}) where T
  res = unit(DILS{T})
  for d in ds
    res = tensor(res, d)
  end
  res
end # reduce(tensor, ds, unit(DILS{T}))

function oapply(d::WiringDiagram, dils::Vector{<:DILS})::DILS
  # check that dils[i] "fills" box i of d
  @assert length(dils) == nboxes(d)
  for (dil, box) in zip(dils, 1:nboxes(d))
    @assert length(input_ports(d, box)) == length(input_polys(dil))
    @assert length(output_ports(d, box)) == length(output_polys(dil))
  end

  layers = layerize(internal_graph(d))

  t_layers = [tensor([dils[i] for i in layer]) for layer in layers]
  res = t_layers[1]
  for (i, (layer, dils_layer)) in enumerate(zip(layers[2:end], t_layers[2:end]))
    weights = hcat(hcat(map(layer) do br # the right layer
      vcat(map(incident(d.diagram, br, :in_port_box)) do pr
        w_pr = incident(d.diagram, pr, :tgt)
        map(layers[i]) do bl
          map(incident(d.diagram, bl, :out_port_box)) do pl
            length(w_pr ∩ incident(d.diagram, pl, :src))
          end
        end
      end...)
    end...)...)
    res = sequential_compose(res, dils_layer, weights)
  end
  return res
end


"""
Partition vertices of a graph by dependency. Get ordered layers.
Can be used for composing morphisms in a SMC.
"""
function layerize(G::AbstractGraph)::Vector{Vector{Int}}
  seen = Set{Int}()
  layers = Vector{Int}[]
  while length(seen) < nv(G)
    next_layer = Int[]
    for v in filter(x->x∉ seen, parts(G, :V))
      if G[incident(G, v, :tgt) , :src] ⊆ seen
        push!(next_layer, v)
      end
    end
    push!(layers, next_layer)
    union!(seen, next_layer)
  end
  return layers
end


"""
Multiply each partition of a i×j partitioned matrix by a scalar, taken from a
i×j scalar matrix.
"""
function generalized_kroenecker(weights::AbstractMatrix, rowdims::Vector{Int}, coldims::Vector{Int})
  blockrow = map(enumerate(coldims)) do (col, coldim)
    block = map(enumerate(rowdims)) do (row, rowdim)
        weights[row,col] * (rowdim == coldim ? I(rowdim) : zeros(rowdim, coldim))
    end
    vcat(block...)
  end
  return hcat(blockrow...)
end



end # module