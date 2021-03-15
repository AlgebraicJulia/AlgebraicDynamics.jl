""" Tonti Diagrams
This file includes a framework for constructing and evaluating Tonti diagrams.
Tonti diagrams are stored as ACSets, and have an imperative interface for
describing physical variables and the relationships between them. This tooling
also lets a Tonti diagram be converted to a vectorfield, allowing for
simulation of physical systems through any traditional vectorfield solver.
"""

module Tonti
using Catlab.Present
using Catlab.Theories
using Catlab.CategoricalAlgebra
using Catlab.CategoricalAlgebra.FinSets
using Catlab.Programs
using Catlab.WiringDiagrams
using CombinatorialSpaces
import CombinatorialSpaces.DualSimplicialSets: EmbeddedDeltaDualComplex2D, ⋆

export TontiDiagram, AbstractTontiDiagram, addTransform!, addSpace!,
       addTime!, vectorfield, disj_union, gen_form, addBC!

# Helper Function for constructing dual complexes
function EmbeddedDeltaDualComplex2D(d::EmbeddedDeltaSet2D{O, D}) where {O, D}
  sd = EmbeddedDeltaDualComplex2D{O, eltype(D), D}(d)
  subdivide_duals!(sd, Barycenter())
  sd
end;

""" Construct a 0-form based on a scalar function

This operator accepts a scalar function and evaulates it at each point on the
simplex, returning a 0-form.
"""
function gen_form(s::EmbeddedDeltaSet2D, f::Function)
  map(f, point(s))
end

# This defines the structure of the Tonti Diagram ACSet
@present TheoryTontiDiagram(FreeSchema) begin
  Func::Data
  Label::Data

  Variable::Ob
  Corner::Ob
  Transform::Ob
  InParam::Ob
  OutParam::Ob

  tgt::Hom(InParam, Transform)
  src::Hom(OutParam, Transform)
  in_var::Hom(InParam, Variable)
  out_var::Hom(OutParam, Variable)

  func::Attr(Transform, Func)
  t_type::Attr(Transform, Label)

  corner::Hom(Variable, Corner)

  c_label::Attr(Corner, Label)
  v_label::Attr(Variable, Label)
end

# These functions are necessary for defining a TontiDiagram data structure from
# the ACSet
const AbstractTontiDiagram = AbstractACSetType(TheoryTontiDiagram)
const TontiDiagram = ACSetType(TheoryTontiDiagram,index=[:src, :tgt, :corner,
                                                         :in_var, :out_var],
                                                 unique_index=[:v_label,
                                                               :c_label])

# Define an interface for an OpenTontiDiagram which allows Tonti diagrams to
# be composed over their corners
const OpenTontiDiagramOb, OpenTontiDiagram = OpenACSetTypes(TontiDiagram,
                                                            :Corner)
Open(td::AbstractTontiDiagram) = begin
  OpenTontiDiagram{Function, Symbol}(td,
                                     FinFunction(collect(1:nparts(td, :Corner)),
                                                 nparts(td, :Corner)))
end

""" Constructor for Tonti Diagram

This constructor generates an empty Tonti diagram of a given dimension. The
dimensionality determines which spatial elements will be included in the
data structure.
"""
function TontiDiagram(dimension = 3)
  td = TontiDiagram{Function, Symbol}()
  corners = ['P', 'L', 'S', 'V']

  for d in 1:(dimension+1)
    d2 = dimension - (d-1) + 1
    primal = add_parts!(td, :Corner, 2, c_label=[Symbol("I$(corners[d])"),
                                                  Symbol("T$(corners[d])")])
    dual = add_parts!(td, :Corner, 2, c_label=[Symbol("I$(corners[d2])2"),
                                                Symbol("T$(corners[d2])2")])
  end
  td
end

""" Constructor for Tonti Diagram with variables

This constructor accepts variables which are each associated with a corner.
"""
function TontiDiagram(dimension::Int, variables::Array{Pair{Symbol, Symbol}})
  td = TontiDiagram(dimension)
  add_parts!(td, :Variable, length(variables), v_label=first.(variables),
             corner=[first(incident(td, s, :c_label)) for s in last.(variables)])
  td
end

# Helper functions for working on Tonti diagrams
dimension(td::AbstractTontiDiagram) = nparts(td, :Corner) ÷ 4 - 1
var_corner(td::AbstractTontiDiagram, label::Symbol) = td[incident(td, label, :v_label), :corner]
get_var_ind(td::AbstractTontiDiagram, label::Symbol) = first(incident(td, label, :v_label))

""" Generates the appropriate hodge star operator between given corners

This operator will produce a scalar identity if the operator is between corners
where a hodge star is no defined.

TODO: This should throw an error instead of failing silently for invalid
inputs. Currently fails silently because the hodge star is added between all
transforms (this needs to be fixed in vectorfield before here)
"""
function gen_star(dom::Int64, codom::Int64, td::AbstractTontiDiagram, dual::EmbeddedDeltaDualComplex2D)
  corner2ind = Dict('P'=>0, 'L'=>1, 'S'=>2, 'V'=>3)
  # Check for which ⋆ operator to apply
  if "$(td[dom, :c_label])"[end] == '2'
    if "$(td[codom, :c_label])"[end] == '2'
      return 1
      #error("Transformation $(td[dom, :c_label]) -> $(td[codom, :c_label]) is not between primal and dual complexes")
    end
    inv(⋆(corner2ind["$(td[codom, :c_label])"[2]], dual))
  else
    if "$(td[codom, :c_label])"[end] != '2'
      return 1
      #error("Transformation $(td[dom, :c_label]) -> $(td[codom, :c_label]) is not between primal and dual complexes")
    end
    ⋆(corner2ind["$(td[dom, :c_label])"[2]], dual)
  end
end

""" Selects appropriate hodge star operator between given corners from a given
pre-computed array

TODO: This should also throw an error, but currently fails silently. See other
gen_star for details.
"""
function gen_star(dom::Int64, codom::Int64, td::AbstractTontiDiagram, star_arr)
  corner2ind = Dict('P'=>0, 'L'=>1, 'S'=>2, 'V'=>3)
  # Check for which ⋆ operator to apply
  if "$(td[dom, :c_label])"[end] == '2'
    if "$(td[codom, :c_label])"[end] == '2'
     return 1
    end
    inv(star_arr[corner2ind["$(td[codom, :c_label])"[2]]+1])
  else
    if "$(td[codom, :c_label])"[end] != '2'
      return 1
    end
    star_arr[corner2ind["$(td[dom, :c_label])"[2]]+1]
  end
end

""" Adds a transformation to the Tonti diagram between given variables

TODO: Remove this function, since hodge-star is not defined element-wise and
      cannot be added at this step.
"""
function addTransform!(td::AbstractTontiDiagram, complex::EmbeddedDeltaSet2D,
                       dom::Array{Symbol}, func::Function,
                       codom::Array{Symbol})
  # Check rules for transforms
  dual = EmbeddedDeltaDualComplex2D(complex)
  dom_sim = var_corner(td, first(dom))
  codom_sim = var_corner(td, first(codom))
  all(var_corner(td, d) == dom_sim for d in dom) || error("Domain is not consistently on the corner $dom_sim")
  all(var_corner(td, c) == codom_sim for c in codom) || error("Codomain is not consistently on the corner $codom_sim")

  star_op = gen_star(dom_sim, codom_sim, td, dual)
  new_func(x...) = begin
    func(x...)
  end
  # Add transformation
  tran = add_part!(td, :Transform, func=new_func, t_type=:constitutive)
  add_parts!(td, :InParam, length(dom), tgt=tran, in_var=[get_var_ind(td, v) for v in dom])
  add_parts!(td, :OutParam, length(codom), src=tran, out_var=[get_var_ind(td, v) for v in codom])
end

""" Adds a transformation to the Tonti diagram between given variables

This expects a Julia function which will take as input an array of values the
length of argument `dom` and will return an array the length of argument
`codom`. The order of variables passed to and recieved from `func` is
determined by the order in `dom` and `codom`.
"""
function addTransform!(td::AbstractTontiDiagram, dom::Array{Symbol}, func::Function, codom::Array{Symbol})
  # Check rules for transforms
  dom_sim = var_corner(td, first(dom))
  codom_sim = var_corner(td, first(codom))
  all(var_corner(td, d) == dom_sim for d in dom) || error("Domain is not consistently on the corner $dom_sim")
  all(var_corner(td, c) == codom_sim for c in codom) || error("Codomain is not consistently on the corner $codom_sim")


  # Add transformation
  tran = add_part!(td, :Transform, func=func, t_type=:constitutive)
  add_parts!(td, :InParam, length(dom), tgt=tran, in_var=[get_var_ind(td, v) for v in dom])
  add_parts!(td, :OutParam, length(codom), src=tran, out_var=[get_var_ind(td, v) for v in codom])
end

""" Adds a topological transformation to the Tonti diagram between given variables

The primary purpose of this function is to ensure that the variables are
appropriate for a topological transformation to exist between them.
"""
function addTopoTransform!(td::AbstractTontiDiagram, dom::Symbol, func::Function, codom::Symbol)
  v_dom = incident(td, dom, [:corner, :c_label])
  v_codom = incident(td, codom, [:corner, :c_label])

  (length(v_dom) <= 1 && length(v_codom) <= 1) ||
      error("Adding topological transformations with >1 variable per corner is not yet supported")

  if length(v_dom) == 0 || length(v_codom) == 0
    return false
  end

  tran = add_part!(td, :Transform, func=func, t_type=:topological)
  add_part!(td, :InParam, tgt=tran, in_var=v_dom[1])
  add_part!(td, :OutParam, src=tran, out_var=v_codom[1])
  return true
end

""" Adds a temporal transformation to the Tonti diagram between given variables

The primary purpose of this function is to ensure that the variables are
appropriate for a temporal transformation to exist between them.
"""
function addTempTransform!(td::AbstractTontiDiagram, dom::Symbol, func::Function, codom::Symbol)
  v_dom = incident(td, dom, [:corner, :c_label])
  v_codom = incident(td, codom, [:corner, :c_label])

  (length(v_dom) <= 1 && length(v_codom) <= 1) ||
      error("Adding topological transformations with >1 variable per corner is not yet supported")

  if length(v_dom) == 0 || length(v_codom) == 0
    return false
  end

  tran = add_part!(td, :Transform, func=func, t_type=:temporal)
  add_part!(td, :InParam, tgt=tran, in_var=v_dom[1])
  add_part!(td, :OutParam, src=tran, out_var=v_codom[1])
  return true
end

""" Adds a boundary condition to a Tonti Diagram

TODO: Extend this to beyond a mask over certain values. Currently this allows
for constant boundaries by applying a mask to the vectorfield result, but this
does not generalize to all mathematical boundary conditions.
"""
function addBC!(td::AbstractTontiDiagram, dom::Symbol, mask::Function)
  v_dom = incident(td, dom, [:v_label])

  tran = add_part!(td, :Transform, func=mask, t_type=:bc)
  add_part!(td, :InParam, tgt=tran, in_var=v_dom[1])
  add_part!(td, :OutParam, src=tran, out_var=v_dom[1])
end

""" Adds topological transformations to a 1D Tonti Diagram

Note that since the complex is not embedded, we are unable to use a true
dual derivative.
"""
function addSpace!(td::AbstractTontiDiagram, complex::AbstractDeltaSet1D)
  bound_1_0   = boundary(1,complex)
  cobound_0_1 = d(0,complex)
  addTopoTransform!(td, :IP, x->(cobound_0_1*x), :IL)
  addTopoTransform!(td, :TP, x->(cobound_0_1*x), :TL)

  # TODO: Add Hodge * operator instead of just swapping bound/cobound
  addTopoTransform!(td, :IP2, x->(bound_1_0*x), :IL2)
  addTopoTransform!(td, :TP2, x->(bound_1_0*x), :TL2)

  td
end

""" Adds topological transformations to a 2D Tonti Diagram
(from an embedded complex)
"""
function addSpace!(td::AbstractTontiDiagram, complex::EmbeddedDeltaSet2D)
  dual = EmbeddedDeltaDualComplex2D(complex)
  d_0_1 = d(0,complex)
  d_1_2 = d(1,complex)

  dd_0_1 = dual_derivative(0, dual)
  dd_1_2 = dual_derivative(1, dual)

  addTopoTransform!(td, :IP, x->(d_0_1*x), :IL)
  addTopoTransform!(td, :TP, x->(d_0_1*x), :TL)
  addTopoTransform!(td, :IL, x->(d_1_2*x), :IS)
  addTopoTransform!(td, :TL, x->(d_1_2*x), :TS)

  # TODO: Add Hodge * operator instead of just swapping bound/cobound
  addTopoTransform!(td, :IP2, x->(dd_0_1*x), :IL2)
  addTopoTransform!(td, :TP2, x->(dd_0_1*x), :TL2)
  addTopoTransform!(td, :IL2, x->(dd_1_2*x), :IS2)
  addTopoTransform!(td, :TL2, x->(dd_1_2*x), :TS2)
  td
end

""" Adds topological transformations to a 2D Tonti Diagram

Note that since the complex is not embedded, we are unable to use a true
dual derivative.

TODO: These should be able to be removed, but considerations must be made
for what it means for a particle system to exist on an embedded complex.
"""
function addSpace!(td::AbstractTontiDiagram, complex::AbstractDeltaSet2D)
  d_0_1 = d(0,complex)
  d_1_2 = d(1,complex)

  dd_0_1 = boundary(2, complex)
  dd_1_2 = boundary(1, complex)

  addTopoTransform!(td, :IP, x->(d_0_1*x), :IL)
  addTopoTransform!(td, :TP, x->(d_0_1*x), :TL)
  addTopoTransform!(td, :IL, x->(d_1_2*x), :IS)
  addTopoTransform!(td, :TL, x->(d_1_2*x), :TS)

  # TODO: Add Hodge * operator instead of just swapping bound/cobound
  addTopoTransform!(td, :IP2, x->(dd_0_1*x), :IL2)
  addTopoTransform!(td, :TP2, x->(dd_0_1*x), :TL2)
  addTopoTransform!(td, :IL2, x->(dd_1_2*x), :IS2)
  addTopoTransform!(td, :TL2, x->(dd_1_2*x), :TS2)

  td
end

""" Adds temporal transformations to a Tonti Diagram

Currently all temporal transformations simply apply Euler's method. When
attached to DifferentialEquations, more complex timestepping can be
accomplished.

TODO: Develop more time-stepping techniques for the Tonti diagram structure
"""
function addTime!(td::AbstractTontiDiagram; dt=1)
  addTempTransform!(td, :TP, x->dt*x, :IP)
  addTempTransform!(td, :TL, x->dt*x, :IL)
  addTempTransform!(td, :TP2, x->dt*x, :IP2)
  addTempTransform!(td, :TL2, x->dt*x, :IL2)

  if dimension(td) > 1
    addTempTransform!(td, :TS, x->dt*x, :IS)
    addTempTransform!(td, :TS2, x->dt*x, :IS2)
  end

  if dimension(td) > 2
    addTempTransform!(td, :TV, x->dt*x, :IV)
    addTempTransform!(td, :TV2, x->dt*x, :IV2)
  end
end

""" Generates a vectorfield for simulation of a system given a Tonti diagram

This function performs a topological sort to determine dependencies between
variables (forgetting the temporal transformations). It then determines the
necessary set of state variables to define the value of each variable for each
time step.

TODO: Convert dependencies to Graph in Catlab and take advantage of the
topological sort tooling from there. Will significantly reduce the size of this
function.
"""
function vectorfield(td::AbstractTontiDiagram, complex::Union{AbstractDeltaSet1D,
                                                              AbstractDeltaSet2D})
  # Initialize hodge-stars if the tonti diagram is embedded
  star_arr = Array{Union{Number,AbstractArray},1}(undef, 3)
  if(has_part(complex, :Point))
    dual = EmbeddedDeltaDualComplex2D(complex)
    star_arr .= [⋆(i, dual) for i in 0:2]
  else
    star_arr .= ones(Int64,3)
  end

  # Define order of evaluation for transformations
  var_deps = [Set(filter(t -> !(td[td[t,:src], :t_type] in [:temporal, :bc]), incident(td, i, :out_var)))
              for i in 1:nparts(td, :Variable)] # deps per variable

  tran_deps = [Set(td[incident(td, t, :src), :in_var])
               for t in 1:nparts(td, :Transform)] # deps per transforms

  var_to_bc = Dict(map(filter(t->(td[t, :t_type] == :bc), collect(1:nparts(td, :Transform)))) do t
                                td[incident(td, t,:src)[1], :out_var] => t
                              end
                             ) # Map from variables to their boundary conditions (if any)
  evaluated = [td[t, :t_type] == :bc for t in 1:nparts(td, :Transform)]
  to_evaluate = Array{Int64, 1}()
  to_eval_vars = Array{Int64, 1}()
  order = Array{Int64, 1}()
  time_vars = td[incident(td, :temporal, [:src, :t_type]), :out_var]

  append!(to_eval_vars, time_vars)
  for v in to_eval_vars
    if v in keys(var_to_bc)
      push!(order, var_to_bc[v])
    end
  end

  # Add any transformations which now have all variables satisfied
  for v in to_eval_vars
    for t in 1:length(tran_deps)
      delete!(tran_deps[t], v)
      if isempty(tran_deps[t]) && !evaluated[t]
        push!(to_evaluate, t)
        evaluated[t] = true
      end
    end
  end
  to_eval_vars = empty(to_eval_vars)

  # Evaluate transforms which have no more unsatisfied variable dependencies
  while !isempty(to_evaluate)
    cur_tran = pop!(to_evaluate)
    push!(order, cur_tran)

    outputs = incident(td, cur_tran, :src)
    out_vars = td[outputs, :out_var]

    # Update variable dependencies based on this transformation's eval
    for out in 1:length(outputs)
      delete!(var_deps[out_vars[out]], outputs[out])
      if isempty(var_deps[out_vars[out]])
        push!(to_eval_vars, out_vars[out])
        if out_vars[out] in keys(var_to_bc)
          push!(order, var_to_bc[out_vars[out]])
        end
      end
    end

    # Add transforms which have all variables satisfied
    for v in to_eval_vars
      for t in 1:length(tran_deps)
        delete!(tran_deps[t], v)
        if isempty(tran_deps[t]) && !evaluated[t]
          push!(to_evaluate, t)
          evaluated[t] = true
        end
      end
    end
    to_eval_vars = empty(to_eval_vars)
  end

  # Initialize memory
  data, v2ind = initData(td, complex)

  # Create maps from tonti diagram variables to the initialized memory
  timevar_to_ind = Dict{Int, Tuple{Int,Int}}()
  for v in 1:length(time_vars)
    cur_corner = td[time_vars[v], :corner]
    data[cur_corner] = vcat(data[cur_corner], zeros(1,size(data[cur_corner])[2]))
    timevar_to_ind[time_vars[v]] = (cur_corner,size(data[cur_corner])[1])
  end

  transforms = Array{Pair{Array{Tuple{Int, Int},1},Array{Tuple{Int, Int},1}},1}()
  masks = Dict{Pair{Int, Int}, Array{Number}}()
  star_op = Array{Union{Number,AbstractArray},1}()

  for t in 1:nparts(td, :Transform)
    in_vars = [(x ∈ time_vars) ? timevar_to_ind[x] : v2ind[x]
               for x in td[incident(td, t, :tgt), :in_var]]
    out_vars = v2ind[td[incident(td, t, :src), :out_var]]
    push!(star_op, gen_star(first(first(in_vars)), first(first(out_vars)), td, star_arr))
    push!(transforms, in_vars => out_vars)
  end

  # Vectorfield function definition
  function system(du, u, t, p)
    # Reset data to 0
    for i in 1:length(data)
      data[i] .= 0
    end

    # Update current state from u
    cur_ind = 1
    for v in time_vars
      c_ind = first(timevar_to_ind[v])
      v_ind = last(timevar_to_ind[v])
      v_len = size(data[c_ind])[2]
      data[c_ind][v_ind,:] .= view(u, cur_ind:(cur_ind+v_len-1))
      cur_ind += v_len
    end

    for t in order
      cur_t = transforms[t]
      dom_c = first(first(first(cur_t)))
      codom_c = first(first(last(cur_t)))
      cur_f = subpart(td, t, :func)
      if subpart(td, t, :t_type) == :topological
        # Topological transforms are a special case (and always 1:1)
        # Will likely want to deal with these uniquely in future implementations
        data[codom_c][last.(cur_t[2]), :] .+= cur_f(data[dom_c][last.(cur_t[1]),:]')'
      elseif subpart(td, t, :t_type) == :bc
        data[codom_c][last.(cur_t[2]), :] .*= cur_f()'
      else
        data[codom_c][last.(cur_t[2]), :] .+= mapslices(subpart(td, t, :func), view(data[dom_c], last.(cur_t[1]), :), dims=1) * star_op[t]
      end
    end
    du .= vcat([data[v2ind[v][1]][v2ind[v][2], :][:] for v in time_vars]...)
  end
  system, [td[v, :v_label] => size(data[v2ind[v][1]])[2] for v in time_vars]
end

# Helper function which intializes the data storage for vectorfield for a 1D
# system
function initData(td::AbstractTontiDiagram, complex::AbstractDeltaSet1D)
  data = Array{Array{Float64, 2}, 1}()
  corner_to_len = Dict(:IP => nv(complex), :TP => nv(complex), :IL2 => nv(complex), :TL2 => nv(complex),
                       :IL => ne(complex), :TL => ne(complex), :IP2 => ne(complex), :TP2 => ne(complex))

  v2ind = fill((0,0), nparts(td, :Variable))
  for c in 1:nparts(td, :Corner)
    cur_vars = incident(td, c, :corner)
    push!(data, zeros(length(cur_vars),corner_to_len[td[c, :c_label]]))
    v2ind[cur_vars] = [(c, v) for v in 1:length(cur_vars)]
  end
  data, v2ind
end

# Helper function which intializes the data storage for vectorfield for a 2D
# system
function initData(td::AbstractTontiDiagram, complex::AbstractDeltaSet2D)
  data = Array{Array{Float64, 2}, 1}()
  c_nv = nv(complex)
  c_ne = ne(complex)
  c_nt = ntriangles(complex)
  corner_to_len = Dict(:IP => c_nv, :TP => c_nv, :IS2 => c_nv, :TS2 => c_nv,
                       :IL => c_ne, :TL => c_ne, :IL2 => c_ne, :TL2 => c_ne,
                       :IS => c_nt, :TS => c_nt, :IP2 => c_nt, :TP2 => c_nt)

  v2ind = fill((0,0), nparts(td, :Variable))
  for c in 1:nparts(td, :Corner)
    cur_vars = incident(td, c, :corner)
    push!(data, zeros(length(cur_vars),corner_to_len[td[c, :c_label]]))
    v2ind[cur_vars] = [(c, v) for v in 1:length(cur_vars)]
  end
  data, v2ind
end


""" Constructs a Tonti diagram from two source Tonti diagrams, combining over
the corners.

The result Tonti diagram contains all information from the independent Tonti
diagrams and with variables on the same corners as before.

TODO: Make this accept a UWD so that tonti diagrams can be joined over shared
variables as well (also just provides a cleaner joining interface)
"""
function disj_union(td1::AbstractTontiDiagram, td2::AbstractTontiDiagram)
  o_td1 = Open(td1)
  o_td2 = Open(td2)

  td_merge = @relation (x,) begin
     td1(x)
     td2(x)
  end

  apex(oapply(td_merge, Dict(:td1=>o_td1, :td2=>o_td2)))
end

end
