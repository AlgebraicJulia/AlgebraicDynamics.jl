module DiscDynam
using Catlab
using Catlab.Programs
using Catlab.Graphics
using Catlab.WiringDiagrams
using Catlab.WiringDiagrams.UndirectedWiringDiagrams
using Catlab.Theories
using Catlab.CategoricalAlgebra

import Catlab.WiringDiagrams.UndirectedWiringDiagrams: TheoryUWD

export TheoryDynamUWD, DynamUWD, AbstractDynamUWD, update!, isconsistent
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
end #module
