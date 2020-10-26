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
  dynamics, dynamics!,
  Dynam, functor, @relation, set_values!

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
DynamUWD() = DynamUWD{Real, Function}()
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
function update!(d::AbstractDynamUWD)
    # Apply the dynamic of each box on its incident states
    boxes = 1:nparts(d, :Box)
    diffs = map(boxes) do b
        states = incident(d, b, :system)
        values = subpart(d, states, :value)
        dynamics = subpart(d, b, :dynamics)
        newvalues = dynamics(values)
        set_subpart!(d, states, :value, newvalues)
        @assert subpart(d, states, :value) == newvalues
        diff = newvalues .- values
    end |> x-> foldl(vcat, x)

    # Apply the cumulative differences to appropriate junctions
    juncs = 1:nparts(d, :Junction)
    map(juncs) do j
        p = incident(d, j, :junction)
        statesp = subpart(d, p, :state)
        nextval = sum(diffs[statesp]) + subpart(d, j, :jvalue)
        set_subpart!(d, j, :jvalue, nextval)
        @assert subpart(d, j, :jvalue) == nextval
        set_subpart!(d, statesp, :value, fill(nextval, length(statesp)))
        @assert all(subpart(d, statesp, :value) .== nextval)
    end
    subpart(d,:, :value)
end

function update!(y::AbstractVector, f::Function, x::AbstractVector)
    y .= f(x)
end

function update!(newstate::AbstractVector, d::AbstractDynamUWD, state::AbstractVector)
    # Apply the dynamic of each box on its incident states
    boxes = 1:nparts(d, :Box)
    for b in boxes
        states = incident(d, b, :system)
        dynamics = subpart(d, b, :dynamics)
        newvalues = update!(view(newstate, states), dynamics, view(state, states))
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

struct Dynam
    dynam::AbstractDynamUWD
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
    Dynam(d_box)
end
dynam(d::Dynam) = subpart(d.dynam, :dynamics)[1]
states(d::Dynam) = nparts(d.dynam, :State)
portmap(d::Dynam) = subpart(d.dynam, :state)
values(d::Dynam) = subpart(d.dynam, :value)

update!(y::AbstractVector, f::Dynam, x::AbstractVector) = update!(y, f.dynam, x)

function functor(transform::Dict{Symbol, Dynam})
  function ob_to_dynam(rel::UntypedRelationDiagram)
    cur_dyn = DynamUWD()
    src = transform[subpart(rel, 1, :name)].dynam
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

function compose(cosp::Cospan)
  function operation(systems::Dynam...)
    res = DynamUWD{Real, Union{Function, Dynam}}()
    # Add system boxes
    add_parts!(res, :Box, length(systems), dynamics=systems)

    # Add junctions
    add_parts!(res, :Junction, length(cosp.apex))

    # Add global outerports
    add_parts!(res, :OuterPort, length(cosp.legs[2].func),
                    outer_junction=cosp.legs[2].func)

    # Add appropriate states and ports
    tot_states = 0
    for (i,sys) in enumerate(systems)
      add_parts!(res, :State, nparts(sys.dynam, :State), system=i, value=subpart(sys.dynam, :value))
      # How do we choose what state is assigned to the outerport?
      # Currently we just choose the first port connected to the junction since this *should* always
      #   be equivalent to other ports on the junction
      add_parts!(res, :Port, nparts(sys.dynam, :OuterPort), box=i,
                      state=tot_states .+ [first(incident(sys.dynam, port, :junction)) for port in subpart(sys.dynam, :outer_junction)])
      tot_states += nparts(sys.dynam, :State)
    end
    set_subpart!(res, :junction, cosp.legs[1].func)
    set_subpart!(res, :jvalue, subpart(res, subpart(res, map(first, incident(res, :, :junction)), :state), :value))
    res
  end
end

function set_values!(d::AbstractDynamUWD{<:Number, Function}, values::Array{<:Number})
    @assert length(values) == nparts(d, :State)
    set_subpart!(d, :value, values)
    set_subpart!(d, subpart(d, :junction), :jvalue,
                    subpart(d, subpart(d, :state), :value))
end
end #module
