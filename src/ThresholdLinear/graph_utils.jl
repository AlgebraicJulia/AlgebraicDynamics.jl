import Base.Iterators: flatten
using MLStyle
using StructEquality

using Catlab
import Catlab: Graph, LabeledGraph, add_edges!, nv, ne, vertices

export nv, ne, Graph

# TODO move
"""    indicator(g::AbstractGraph, σ::Vector{Int})

Get the indicator function of a subset with respect to a graph. Returns a vector of 0 and 1s.

This should probably be a FinFunction.
"""
function indicator(g::AbstractGraph, σ::Vector{Int})
    [Int(v ∈ σ) for v ∈ vertices(g)]
end
export indicator

shift(g::Graph, args...) = g
# XXX When making binary combinations of graphs, we first create a disjoint union of two graphs and then shift the vertex IDs of the right graph so that the new edges have proper targets. This is handled in the code for all such "union" operations, but we have a helper method called `shift` for handling the implicit graph cases and any future case where combinatorial structures might have attributes which may be shifted. In other words, we shift to avoid colliding vertex labels. Plain graphs do not have labels, as they do not have attributes at all, so shifting it does nothing at all.

""" An implicit cyclic graph with `n` vertices. The `offset` of an ImplicitGraph tracks the first vertex starts """
abstract type ImplicitGraph end

@struct_hash_equal mutable struct CycleGraph <: ImplicitGraph
    n::Int
    offset::Int
    CycleGraph(n::Int, offset=0) = new(n, offset)
end
export CycleGraph

Graph(g::CycleGraph) = cycle_graph(Graph, nv(g))

nv(g::CycleGraph) = g.n 
ne(g::CycleGraph) = g.n 

Base.copy(g::CycleGraph) = CycleGraph(g.n, g.offset)

@struct_hash_equal mutable struct CompleteGraph <: ImplicitGraph
    n::Int
    offset::Int
    CompleteGraph(n::Int, offset=0) = new(n, offset)
end
export CompleteGraph

Graph(g::CompleteGraph) = complete_graph(Graph, nv(g))

nv(g::CompleteGraph) = g.n 
ne(g::CompleteGraph) = g.n^2

Base.copy(g::CompleteGraph) = CompleteGraph(g.n, g.offset)

@struct_hash_equal mutable struct DiscreteGraph <: ImplicitGraph
    n::Int
    offset::Int
    DiscreteGraph(n::Int, offset=0) = new(n, offset)
end
export DiscreteGraph

Graph(g::DiscreteGraph) = Graph(nv(g))

nv(g::DiscreteGraph) = g.n 
ne(g::DiscreteGraph) = 0

Base.copy(g::DiscreteGraph) = DiscreteGraph(g.n, g.offset)

function shift!(g::T, n) where T <: ImplicitGraph
    g.offset += n
    g
end
shift(g::T, n) where T <: ImplicitGraph = shift!(copy(g), n)
export shift, shift!

vertices(g::ImplicitGraph) = 1:nv(g)

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

function clique_union(G::Graph, H::ImplicitGraph)::Graph
    clique_union(G, Graph(H))
end

clique_union(H::ImplicitGraph, G::Graph) = clique_union(G, H)

function clique_union(args...)::Graph
    @assert eltype(args) == Graph
    reduce(clique_union, args)
end

# TODO the right thing to do here is promote this to a DisjointUnion, since the arguments are assumed to be Lazy
function Base.:+(g::ImplicitGraph, h::ImplicitGraph)
    disjoint_union(Graph(g), Graph(h))
end

# forward chain
function chain_union(G::Graph, H::Graph; offset::Int=0)::Graph
    X = disjoint_union(G, H)
    g_offset = nv(G)
    for v in vertices(G)
        v > offset || continue
        for w in vertices(H)
            add_edge!(X, v, w + g_offset)
        end
    end
    X
end

function cyclic_union(G::Graph, H::Graph)::Graph
    cyclic_union([G, H])
end

# [G1, G2, G3, G4]
# g12 <- G1.[v >= 1] => G2
# g23 <- (g12).[v >= offsets[|G1|]] => G3
# g34 <- (g23).[v >= offsets[|G2|]] => G4...
function cyclic_union(graphs::Vector{Graph})::Graph
    nvs = [0; accumulate(+, nv.(graphs))...]
    chain = foldl(enumerate(graphs)) do (idx, fst), (_, snd)
        chain_union(fst, snd; offset=nvs[idx])
    end
    # loop back around
    for v in vertices(chain)
        v > nvs[end-1] || continue
        for w in vertices(graphs[1])
            add_edge!(chain, v, w)
        end
    end
    chain
end
# \circlearrowright

function cyclic_union(G::Graph, H::ImplicitGraph)::Graph
    cyclic_union(G, Graph(H))
end

function cyclic_union(H::ImplicitGraph, G::Graph)::Graph
    cyclic_union(G, H)
end

###


"""
    connected_union(G::Graph, H::Graph) -> Graph

Construct the **connected union** of two graphs `G` and `H`.

This operation forms the set-theoretic union of the vertex and edge sets of
`G` and `H`. Vertices with the same labels are identified (no relabeling or
index shifting is performed prior to taking the union), so shared labels make
the result “connected” through common nodes.

# Example
```julia
julia> G = CycleGraph(5)
julia> H = CycleGraph(3)
julia> X = connected_union(G, H)
julia> nv(X), ne(X)
(5, 5)
```
ImplicitGraph arguments are first realized as explicit Graph objects
and then unioned.
"""
function connected_union(G::Graph, H::Graph)
    # 1) union of vertex labels
    L = maximum([vertices(G), vertices(H)])
    # 2) unpack and edge e as a pair (src(e), tgt(e))
    δ(G, e) = (src(G, e), tgt(G, e))
    # 3) base graph with |L| vertices
    X = Graph(length(L))
    # build the set of all edges from g and h, wrtten as pairs (src(e), tgt(e))
    E = Set([δ.(Ref(G), edges(G))..., δ.(Ref(H), edges(H))...])
    # 4) add edges to X
    for(s, t) in E
        add_edge!(X, s, t)
    end
    return X
end

function connected_union(G::Graph, H::ImplicitGraph)
    connected_union(G, Graph(H))
end

function connected_union(G::ImplicitGraph, H::Graph)
    connected_union(Graph(G), H)
end

function connected_union(G::ImplicitGraph, H::ImplicitGraph)
    connected_union(Graph(G), Graph(H))
end

export connected_union

