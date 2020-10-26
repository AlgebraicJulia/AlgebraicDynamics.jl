module Functors

using Catlab
using Catlab.Theories
using Catlab.Programs.RelationalPrograms
using Catlab.CategoricalAlgebra
using Catlab.CategoricalAlgebra.FinSets
using Catlab.CategoricalAlgebra.CSetDataStructures
using Catlab.CategoricalAlgebra.StructuredCospans

export Functor, compose, split!

struct Functor
  ob
  comp
  split
end

function (F::Functor)(uwd::RelationDiagram)
  _functor(F, copy(uwd))
end

function _functor(F::Functor, uwd::RelationDiagram)
  # Base Case: UWD of 1 box
  if nparts(uwd, :Box) == 1
      return F.ob(uwd)
  end
  # TODO: Make a more generalized version so that we don't have
  # to arbitrarily keep track of OuterPort order

  # Split box, apply functor, then re-compose
  op, cosp, l_uwd, r_uwd = F.split(uwd)
  return F.comp(op, cosp, _functor(F, l_uwd), _functor(F, r_uwd))
end

# TODO: Add some kind of checking to ensure that `ob` returns valid objects
function Functor(ob, OpenType)
  Functor(ob, gen_compose(OpenType), split!)
end
function Functor(ob)
  Functor(ob, gen_compose(UntypedRelationDiagram), split!)
end


# Generic compose function that should
function gen_compose(OpenType)
  OpenTOb, OpenT = OpenACSetTypes(OpenType, :Junction)
  function typedComp(op_map::Array{Int, 1}, cosp::Cospan, a::T, b::T) where {CD, AD, Ts, T <: ACSet{CD, AD, Ts}}
    # Get inverses of the cospan legs to convert it to a span
    a_inv = zeros(Int, length(cosp.apex))
    a_inv[cosp.legs[1].func] = 1:length(cosp.legs[1].func)

    b_inv = zeros(Int, length(cosp.apex))
    b_inv[cosp.legs[2].func] = 1:length(cosp.legs[2].func)

    # Generate boundaries, and fill in junctions necessary to convert cospan to span
    for i in 1:length(cosp.apex)
      if a_inv[i] == 0 && b_inv[i] == 0
        throw(error("Index $i in the apex has no value mapped to it"))
      elseif b_inv[i] == 0
        a_attr = a.tables.Junction[a_inv[i]]
        b_junc = add_part!(b, :Junction, a_attr)
        b_inv[i] = b_junc
      elseif a_inv[i] == 0
        b_attr = b.tables.Junction[b_inv[i]]
        a_junc = add_part!(a, :Junction, b_attr)
        a_inv[i] = a_junc
      end
    end

    # We now have sufficient data on both ends to use open composition
    open_a = OpenT{Ts.parameters...}(a, FinFunction(Array{Int, 1}(), nparts(a, :Junction)), FinFunction(a_inv, nparts(a, :Junction)))
    open_b = OpenT{Ts.parameters...}(b, FinFunction(b_inv, nparts(b, :Junction)), FinFunction(Array{Int, 1}(), nparts(b, :Junction)))

    # It might help if this structure stored what objects in a,b map to objects in ab
    ab = compose(open_a, open_b).cospan.apex

    # Bring the resulting junctions back to original order
    reorder_part!(ab, :Junction, a_inv)
    reorder_part!(ab, :OuterPort, op_map)
    return ab
  end
  return typedComp
end

# Splits off one box from a RelationDiagram
function split!(left::UntypedRelationDiagram)
  njuncs = nparts(left, :Junction)
  left_juncs = collect(1:njuncs)
  # Create new RelDiag w/ last right
  right_ind = nparts(left, :Box)
  right = UntypedRelationDiagram{Symbol, Symbol}()
  add_part!(right, :Box, name=subpart(left, right_ind, :name))

  ports = incident(left, right_ind, :box)
  right_juncs = unique(subpart(left, ports, :junction))
  left_to_right = zeros(Int, nparts(left, :Junction))
  left_to_right[right_juncs] = 1:length(right_juncs)
  add_parts!(right, :Junction, length(right_juncs),
                  variable=subpart(left, right_juncs, :variable))
  add_parts!(right, :Port, length(ports),
                  box=fill(1, length(ports)),
                  junction=left_to_right[subpart(left, ports, :junction)])

  # Remove last box
  rem_parts!(left, :Port, ports)
  rem_part!(left, :Box, nparts(left, :Box))

  # Remove any right_juncs from right1 that have no connections
  junctions = filter(j -> isempty(incident(left, j, :junction)), right_juncs)
  # Transfer outer_ports with these
  outerports = vcat(incident(left, junctions, :outer_junction)...)
  left_outer = collect(1:nparts(left, :OuterPort))
  right_outer = Array{Int, 1}()
  reverse!(sort!(outerports)) # Delete from back to front

  # Take care of order of OuterPorts
  left_op_juncs = left.tables.OuterPort[outerports]

  for (i,op) in enumerate(outerports)
    add_part!(right, :OuterPort)
    for k in keys(left_op_juncs[i])
      v = left_op_juncs[i][k]
      if k == :outer_junction
        set_subpart!(right, i, k, left_to_right[v])
      else
        set_subpart!(right, i, k, v)
      end
    end
    rem_part!(left, :OuterPort, op)
    push!(right_outer, left_outer[op])
    left_outer[op] = left_outer[end]
    pop!(left_outer)
  end

  # Remove unnecessary junctions
  reverse!(sort!(junctions))

  # NOTE: This section is dependent upon the specific deletion algorithm used by
  for j in junctions
    rem_part!(left, :Junction, j)
    left_juncs[j] = left_juncs[end]
    pop!(left_juncs)
  end
  cosp = Cospan(FinFunction(left_juncs, njuncs), FinFunction(right_juncs, njuncs))
  vcat(left_outer, right_outer), cosp, left, right
end

# Provide a new order for a given part of the ACSet
# `order` is assumed to be a bijection. Maybe make this into
# a generated function and upstream to Catlab?
reorder_part!(acs::ACSet, type::Symbol, order::Array{Int, 1}) =
  _reorder_part!(acs, Val(type), order)

function _reorder_part!(acs::ACSet{CD,AD,Ts,Idxed}, ::Val{ob},
                               order::Array{Int, 1}) where {CD,AD,Ts,Idxed,ob}
  in_homs = filter(hom -> codom(CD, hom) == ob, CD.hom)
  indexed_out_homs = filter(hom -> dom(CD, hom) == ob && hom ∈ Idxed, CD.hom)
  indexed_attrs = filter(attr -> dom(AD, attr) == ob && attr ∈ Idxed, AD.attr)
  last_part = length(acs.tables[ob])


  # Check for bijection
  @assert length(order) == last_part
  @assert maximum(order) == last_part
  @assert minimum(order) == 1
  @assert length(unique(order)) == last_part

  order_adj = zeros(Int, last_part)
  order_adj[order] = collect(1:last_part)

  # Swap hom references
  for hom in Tuple(in_homs)
    incident_obs = [incident(acs, i, hom, copy=true) for i in 1:last_part]
    for i in order
      set_subpart!(acs, incident_obs[i], hom, order[i])
    end
  end

  # Swap
  junc_attrs = acs.tables[ob]
  junc_dict = Dict(map(keys(junc_attrs[1])) do k
                            (k, [junc_attrs[i][k] for i in order_adj])
                            end)
  for attr in Tuple(indexed_attrs)
    for i in 1:last_part
        unset_data_index!(acs.indices[attr], junc_attrs[i][attr], i)
    end
  end
  set_subparts!(acs, 1:last_part, junc_dict)
end

# Copied in from CSetDataStructures in Catlab
function unset_data_index!(d::AbstractDict{K,Int}, k::K, v::Int) where K
  if haskey(d, k) && d[k] == v
    delete!(d, k)
  end
end
function unset_data_index!(d::AbstractDict{K,<:AbstractVector{Int}},
                           k::K, v::Int) where K
  if haskey(d, k)
    vs = d[k]
    if deletesorted!(vs, v) && isempty(vs)
      delete!(d, k)
    end
  end
end

end #module
