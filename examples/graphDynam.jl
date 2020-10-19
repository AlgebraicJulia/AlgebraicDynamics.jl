module GraphDynam
using Catlab
using Catlab.Programs
using Catlab.Graphics
using Catlab.Theories
using Catlab.CategoricalAlgebra

export TheoryGraph, AbstractGraph, Graph

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



# g = Graph()
# add_parts!(g, :Vertex, 4)
# add_parts!(g, :Edge,   5,  src=[1,1,1,2,2], tgt=[2,3,3,2,3])
# @show DFS(g, 3)

d = cyclicDynam(4)
#add_parts!(d, :State, 4, next=[2,3,1,1])
@show g = DynamToGraph(d)
@show DFS(g,1)

g = cyclicGraph(10)
@show DFS(g, 2)


end #module