module DiscDynam
using Catlab
using Catlab.Programs
using Catlab.Graphics
using Catlab.WiringDiagrams
using Catlab.WiringDiagrams.UndirectedWiringDiagrams: AbstractUWD
using Catlab.Theories
using Catlab.CategoricalAlgebra
import Catlab.CategoricalAlgebra.FinSets: Cospan, FinFunction
using ..Functors

import Catlab.Programs.RelationalPrograms: @relation
import Catlab.WiringDiagrams.UndirectedWiringDiagrams: TheoryUWD

export TheoryDynamUWD, DynamUWD, AbstractDynamUWD, Cospan, FinFunction,
  update!, isconsistent, compose,
  Dynam, dynamics, dynamics!,
  functor, @relation, set_values!,
  get_exposed

# @present TheoryUWD(FreeSchema) begin
#   Box::Ob
#   Port::Ob
#   OuterPort::Ob
#   Junction::Ob

#   box::Hom(Port,Box)
#   junction::Hom(Port,Junction)
#   outer_junction::Hom(OuterPort,Junction)
# end

@present TheoryDynamUWD <: TheoryUWD begin
    State::Ob
    Scalar::Data
    Dynamics::Data

    system::Hom(State, Box)
    state::Hom(Port, State)
    value::Attr(State, Scalar)
    jvalue::Attr(Junction, Scalar)
    #junction ⋅ jvalue == state ⋅ value

    dynamics::Attr(Box, Dynamics)
    # h::Attr(Box, Scalar)
end

const AbstractDynamUWD = AbstractACSetType(TheoryDynamUWD)
const DynamUWD = ACSetType(TheoryDynamUWD, index=[:box, :junction, :outer_junction, :state, :system])
DynamUWD() = DynamUWD{Real, Union{Function, DynamUWD}}()
"""    isconsistent(d::AbstractDynamUWD)

check that all the states associated to ports that are connected to junctions, have the same value as the value of the junction.

As an equation:

    junction⋅jvalue == state⋅value

"""
function isconsistent(d::AbstractDynamUWD)
    jconsistent = map(1:nparts(d, :Junction)) do j
        ports = incident(d, j, :junction)
        states = subpart(d, ports, :state)
        sv = subpart(d, states, :value)
        jv = subpart(d, j, :jvalue)
        all(sv .== jv)
    end
    all(jconsistent)
end

"""    update!(d::AbstractDynamUWD)

compute the new state of a (continuous space) discrete time dynamical system using in-place operations
"""
function update!(d::AbstractDynamUWD, args...)

  input = subpart(d, :value)
  output = zero(input)
  update!(output, d, input, args...)
  set_values!(d, output)
  return output
end

function update!(y::AbstractVector, f::Function, x::AbstractVector, args...)
    y .= f(x, args...)
end

function update!(newstate::AbstractVector, d::AbstractDynamUWD, state::AbstractVector, args...)
    # Apply the dynamic of each box on its incident states
    boxes = 1:nparts(d, :Box)
    for b in boxes
        states = incident(d, b, :system)
        dynamics = subpart(d, b, :dynamics)
        newvalues = update!(view(newstate, states), dynamics, view(state, states), args...)
    end

    # Apply the cumulative differences to appropriate junctions

    juncs = 1:nparts(d, :Junction)
    for j in juncs
        p = incident(d, j, :junction)
        length(p) > 0 || continue
        statesp = subpart(d, p, :state)
        nextval = state[first(statesp)] + mapreduce(i->newstate[i]-state[i], +, statesp, init=0)
        newstate[statesp] .= nextval
    end
    return newstate
end

function dynamics(d::DynamUWD)
    f(x) = begin
        set_subpart!(d, :, :value, x)
        set_subpart!(d, :, :jvalue, x[subpart(d, map(first, incident(d, :, :junction)), :state)])
        # subpart(d, :, :value)
        x′ = update!(d)
        @assert all(x′ .== subpart(d,:,:value))
        return x′
    end
end

function dynamics!(diag::DynamUWD)
    storage = zeros(Float64, nparts(diag, :State))
    return state -> update!(storage, diag, state)
end

function Dynam(dynam::Function, states::Int, portmap::Array{Int,1}, values::Array{<:Real,1})
    @assert maximum(portmap) <= states
    @assert states == length(values)
    d_box = DynamUWD()
    b_ind = add_part!(d_box, :Box, dynamics=dynam)
    st_ind = add_parts!(d_box, :State, states, value=values,
                        system=ones(Int,length(values)))
    jn_ind = add_parts!(d_box, :Junction, length(portmap))
    pt_ind = add_parts!(d_box, :Port, length(portmap), box=ones(Int, length(portmap)),
                                junction=collect(1:length(portmap)),
                                state=portmap)
    set_subpart!(d_box, :jvalue, subpart(d_box, subpart(d_box, incident(d_box, 1:length(portmap), :junction)[1], :state), :value))
    add_parts!(d_box, :OuterPort, length(portmap), outer_junction=1:length(portmap))
    d_box
end
dynam(d::DynamUWD) = subpart(d, :dynamics)[1]
states(d::DynamUWD) = nparts(d, :State)
portmap(d::DynamUWD) = subpart(d, :state)
values(d::DynamUWD) = subpart(d, :value)

function functor(transform::Dict{Symbol, <:AbstractDynamUWD})
  function ob_to_dynam(rel::UntypedRelationDiagram)
    cur_dyn = DynamUWD()
    src = transform[subpart(rel, 1, :name)]
    copy_parts!(cur_dyn, src, (Box=:, Port=:, State=:))
    add_parts!(cur_dyn, :Junction, nparts(rel, :Junction))
    p_to_junc = subpart(rel, :junction)
    set_subpart!(cur_dyn, :junction, p_to_junc)
    set_subpart!(cur_dyn, :jvalue,
                subpart(src, [incident(cur_dyn, i, :junction)[1] for i in 1:nparts(rel, :Junction)], :jvalue))
    add_parts!(cur_dyn, :OuterPort, nparts(rel, :OuterPort), outer_junction=subpart(rel, :outer_junction))
    cur_dyn
  end
  Functor(ob_to_dynam, DynamUWD)
end

# Can either compose this by using heirarchical definition
# Or can keep everything expanded out

# How about a heriarchical definition here, then we can have a "flatten" function called on a Dynam?

function compose(M::Cospan, systems::DynamUWD...)
  # Generate product and cospan of systems
  sys_prod =  coproduct([sys for sys in systems]).cocone.apex
  sys_left =  FinFunction(subpart(sys_prod, :junction),       nparts(sys_prod, :Junction))
  sys_right = FinFunction(subpart(sys_prod, :outer_junction), nparts(sys_prod, :Junction))
  N = Cospan(sys_left, sys_right)

  # Compose cospans
  comp = compose(N, M)

  dyn = DynamUWD()
  # Add in junctions
  inv_comp = zeros(Int, length(comp.apex))
  inv_comp[left(comp).func] = 1:length(left(comp).func)
  for i in 1:length(comp.apex)
    add_part!(dyn, :Junction, sys_prod.tables.Junction[inv_comp[i]])
  end

  # Add other objects
  copy_parts!(dyn, sys_prod, (Box=:, State=:))
  add_parts!(dyn, :Port, nparts(sys_prod, :Port),
                  box=subpart(sys_prod, :box),
                  junction=left(comp).func[subpart(sys_prod, :junction)],
                  state=subpart(sys_prod, :state))
  add_parts!(dyn, :OuterPort, length(right(M).func),
                  outer_junction=right(comp).func[right(M).func])

  set_subpart!(dyn, subpart(dyn, :junction), :jvalue,
                  subpart(dyn, subpart(dyn, :state), :value))
  dyn
end

# Curried form of compose
function compose(cosp::Cospan)
  function operation(systems::DynamUWD...)
    compose(cosp, systems...)
  end
end

function compose(M::Cospan, N::Cospan)
  ιM, ιN = colim = pushout(right(M), left(N))
  cospan = Cospan(ob(colim), ιM, ιN)
end

function set_values!(d::AbstractDynamUWD{T}, values::Array{<:T}) where T
    @assert length(values) == nparts(d, :State)
    set_subpart!(d, :value, values)
    set_subpart!(d, subpart(d, :junction), :jvalue,
                    subpart(d, subpart(d, :state), :value))
end

""" get_exposed(d::AbstractDynamUWD, values::Array)

Given values of the internal states of the dynam d, 
returns the values exposed on the outer_junctions.
"""
function get_exposed(d::AbstractDynamUWD, values::Array)
  jvalues = view(values, subpart(d, map(first, incident(d, parts(d, :Junction), :junction)), :state))
  return view(jvalues, subpart(d, :outer_junction))
end

end #module