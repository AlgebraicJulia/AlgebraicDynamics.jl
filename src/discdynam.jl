module DiscDynam
using Catlab
using Catlab.Programs
using Catlab.Graphics
using Catlab.WiringDiagrams
using Catlab.WiringDiagrams.UndirectedWiringDiagrams: AbstractUWD
using Catlab.Theories
using Catlab.CategoricalAlgebra

import Catlab.Programs.RelationalPrograms: @relation
import Catlab.WiringDiagrams.UndirectedWiringDiagrams: TheoryUWD

export TheoryDynamUWD, DynamUWD, AbstractDynamUWD, update!, isconsistent, Dynam, functor, @relation, set_values!
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

struct Dynam
    dynam::Function
    states::Int
    portmap::Array{Int}
    values::Array{Real}
    function Dynam(dynam::Function, states::Int, portmap::Array{Int,1}, values::Array{<:Real,1})
        @assert maximum(portmap) <= states
        @assert states == length(values)
        new(dynam, states, portmap, values)
    end
end

function functor(transform::Dict{Symbol, Dynam})
    convert(uwd::AbstractUWD) = begin
        dst = DynamUWD()
        add_parts!(dst, :Junction, nparts(uwd, :Junction))
        add_parts!(dst, :Box, nparts(uwd, :Box))
        add_parts!(dst, :Port, nparts(uwd, :Port), box=subpart(uwd, :box),
                                                   junction=subpart(uwd, :junction))
        add_parts!(dst, :OuterPort, nparts(uwd, :OuterPort),
                                    outer_junction=subpart(uwd, :outer_junction))
        for b in 1:nparts(dst, :Box)
            name = subpart(uwd, b, :name)
            ports = incident(dst, b, :box)
            dynam = transform[name]
            set_subpart!(dst, b, :dynamics, dynam.dynam)
            states = add_parts!(dst, :State, dynam.states, system=fill(b, dynam.states),
                                                           value=dynam.values)
            set_subpart!(dst, ports, :state, states[dynam.portmap])
        end
        for p in 1:nparts(dst, :Port)
            j = subpart(dst, p, :junction)
            set_subpart!(dst, j, :jvalue, subpart(dst,subpart(dst, p, :state), :value))
        end
        return dst
    end
    return convert
end

function set_values!(d::AbstractDynamUWD{<:Number, Function}, values::Array{<:Number})
    @assert length(values) == nparts(d, :State)
    set_subpart!(d, :value, values)
    set_subpart!(d, subpart(d, :junction), :jvalue,
                    subpart(d, subpart(d, :state), :value))
end
end #module
