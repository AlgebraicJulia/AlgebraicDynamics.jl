module GameOfLife

using Catlab
using Catlab.Programs
using Catlab.Graphics
using Catlab.Theories
using AlgebraicDynamics
using AlgebraicDynamics.GraphDynam
using AlgebraicDynamics.Machine

# GRAPH DYNAM 

@present TheoryGraph(FreeSchema) begin
    Vertex::Ob
    Edge::Ob

    src::Hom(Edge, Vertex)
    tgt::Hom(Edge, Vertex)
end

const AbstractGraph = AbstractACSetType(TheoryGraph)
const Graph = ACSetType(TheoryGraph, index=[:src, :tgt])

@present TheoryDynam(FreeSchema) begin
    State::Ob
    next::Hom(State, State)
end

const AbstractDynam = AbstractACSetType(TheoryDynam)
const Dynam = ACSetType(TheoryDynam, index=[:next])

function DFS(g::AbstractGraph, start::Int)
    vertices = 1:nparts(g, :Vertex)
    @assert start in vertices
    q = Vector{Int}()
    discovered = Vector{Int}()

    push!(q, start)

    while ! isempty(q)
        v = pop!(q)
        if !(v in discovered)
            push!(discovered, v)
            edges = incident(g, v, :src)
            append!(q, subpart(g, edges, :tgt))
        end
    end

    return discovered

end


function DynamToGraph(d::AbstractDynam)
    g = Graph()
    add_parts!(g, :Vertex, nparts(d, :State))
    add_parts!(g, :Edge  , nparts(d, :State), src=1:nparts(d, :State), tgt=subpart(d,:next))
    return g
end

function cyclicDynam(n::Int)
    d = Dynam()
    nexts = map(1:n) do i
        (i % n) + 1
    end
    add_parts!(d, :State, n, next=nexts)
    return d
end

function cyclicGraph(n::Int)
    return DynamToGraph(cyclicDynam(n))
end

function linearGraph(n::Int)
    g = Graph()
    add_parts!(g, :Vertex,  n)
    add_parts!(g, :Edge,    n-1, src=1:(n-1),   tgt=2:n)
    add_parts!(g, :Edge,    n-1, src=2:n,       tgt=1:(n-1))
    return g
end


@present TheoryValuedGraph <: TheoryGraph begin
    Value::Data 
    value::Attr(Vertex, Value)
end

const AbstractValuedGraph = AbstractACSetType(TheoryValuedGraph)
const ValuedGraph = ACSetType(TheoryValuedGraph, index=[:src, :tgt, :value])

# MACHINES 
@present TheoryDynamMachine(FreeSchema) begin
    Box::Ob
    InPort::Ob
    OutPort::Ob
    State::Ob
    
    Value::Data
    Dynamics::Data

    parameterizes::Hom(InPort, Box)
    state::Hom(OutPort, State)
    system::Hom(State, Box)
    feeder::Hom(InPort, OutPort)
    
    value::Attr(State, Value)
    dynamics::Attr(Box, Dynamics)
end

const AbstractDynamMachine = AbstractACSetType(TheoryDynamMachine)
const DynamMachine = ACSetType(TheoryDynamMachine, index=[:parameterizes, :state, :system, :value, :feeder, :dynamics])
DynamMachine() = DynamMachine{Real, Function}()

@present TheoryDynamEquippedGraph <: TheoryGraph begin
    Dynamics::Data
    dynamics::Attr(Vertex, Dynamics)
end



function update!(dm::AbstractDynamMachine) 
    boxes = 1:nparts(dm, :Box)
    newvalues = map(boxes) do b
        ivalues = subpart(dm, subpart(dm, incident(dm, b, :parameterizes), :feeder), :value)
        xvalues = subpart(dm, incident(dm, b, :system), :value)
        dynamics = subpart(dm, b, :dynamics)
        return dynamics(ivalues, xvalues)
    end

    map(boxes, newvalues) do b, nv
        set_subpart!(dm, incident(dm, b, :system), :value, nv)
    end
end


## Takes a dynamics-valued graph and turns it into a machine
function (dvg::AbstractValuedGraph)
    dm = DynamMachine()

    V = nparts(dvg, :Vertex)
    E = nparts(dm, :Edge)

    add_parts!(dm, :Box,    V,  dynamics=subpart(dvg, 1:V, :value))
    add_parts!(dm, :State,  V,  system=1:V)
    add_parts!(dm, :OutPort,E,  state=subpart(dvg, 1:E, :src))
    add_parts!(dm, :InPort, E,  feeder=1:E, parameterizes=subpart(dvg, 1:E, :tgt))

    return dm


end


# takes a dynamics-valued graph and returns the corresponding machine

end