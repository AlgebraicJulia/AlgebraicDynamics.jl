import Base.Iterators: flatten
using Catlab
import Catlab: Graph, LabeledGraph, add_edges!, nv, ne
using MLStyle

# TODO upstream to Catlab graphs
C(n::Int) = cycle_graph(Graph, n)
K(n::Int) = complete_graph(Graph, n)
D(n::Int) = Graph(n)
export C, K, D

abstract type ImplicitGraph end

mutable struct CycleGraph <: ImplicitGraph
    n::Int
    offset::Int
    CycleGraph(n::Int, offset=0) = new(n, offset)
end
export CycleGraph

Graph(g::CycleGraph) = C(nv(g))

nv(g::CycleGraph) = g.n 
ne(g::CycleGraph) = g.n 

mutable struct CompleteGraph <: ImplicitGraph
    n::Int
    offset::Int
    CompleteGraph(n::Int, offset=0) = new(n, offset)
end
export CompleteGraph

Graph(g::CompleteGraph) = K(nv(g))

nv(g::CompleteGraph) = g.n 
ne(g::CompleteGraph) = g.n^2

mutable struct DiscreteGraph <: ImplicitGraph
    n::Int
    offset::Int
    DiscreteGraph(n::Int, offset=0) = new(n, offset)
end
export DiscreteGraph

Graph(g::DiscreteGraph) = D(nv(g))

nv(g::DiscreteGraph) = g.n 
ne(g::DiscreteGraph) = 0

function shift(g::T, n) where T <: ImplicitGraph
    g.offset += n
    g
end

function shift!(g::T, n) where T <: ImplicitGraph
    g.offset += n
end
export shift!

function LabeledGraph(g::T) where T <: ImplicitGraph
    X = Graph(T(g))
    L = LabeledGraph{Int}(nv(g))
    for e in edges(X) 
        # TODO would be nice to treat the implict graph like an iterator and generate the edges rather than materialize it as an "explicit" Graph first
        add_part!(L, :E, src=g[e, :src], tgt=g[e, :tgt])
    end
    for v in vertices(L)
        set_subpart!(L, v, :label, v + g.offset)
    end
    return L
end
export LabeledGraph

LabeledGraph(::Type{T}, args...) where T <: ImplicitGraph =
    LabeledGraph(T(args...))

# TODO defined for LabeledGraph

"""    The concatenation of two graphs is their union with index shifting. This operation is symmetric up to relabeling. The matrix representation is block concatenation.
"""
function disjoint_union(G::Graph, H::Graph)::Graph
    X = deepcopy(G)
    vs = add_parts!(X, :V, nv(H))
    offset = vs[1] - 1 # TODO why offset by one?
    for e in edges(H)
        add_edge!(X, src(H, e) + offset, tgt(H, e) + offset)
    end
    X
end
export disjoint_union

function disjoint_union(G::Graph, H::ImplicitGraph)::Graph
    disjoint_union(G, Graph(H))
end

disjoint_union(H::ImplicitGraph, G::Graph) = disjoint_union(G, H)

function disjoint_union(args...)::Graph
    @assert eltype(args) == Graph
    reduce(disjoint_union, args)
end

"""    The clique union of two graphs is their disjoint union with an additional bidirectional, bicomplete graph. The matrix representation is block concatenation with a fully-connected matrices in the B and C cells.
"""
function clique_union(G::Graph, H::Graph)::Graph
    X = disjoint_union(G, H)
    offset = nv(G)
    for v in vertices(G)
        for w in vertices(H)
            add_edge!(X, v, w + offset)
            add_edge!(X, w + offset, v)
        end
    end
    X
end
export clique_union

function clique_union(G::Graph, H::ImplicitGraph)::Graph
    clique_union(G, Graph(H))
end

clique_union(H::ImplicitGraph, G::Graph) = clique_union(G, H)

function clique_union(args...)::Graph
    @assert eltype(args) == Graph
    reduce(clique_union, args)
end

# TODO Move me!
abstract type GraphArchitecture end

# TODO the right thing to do here is promote this to a DisjointUnion, since the arguments are assumed to be Lazy
function Base.:+(g::ImplicitGraph, h::ImplicitGraph)
    disjoint_union(Graph(g), Graph(h))
end
