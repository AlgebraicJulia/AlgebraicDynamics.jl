# using AlgebraicDynamics

using MLStyle
using Catlab
using Catlab.Graphics.Graphviz

struct OrderedPartition end

# --------------------------------------

see(g) = (view_graphviz ∘ to_graphviz)(g)

# make a graph
h = star_graph(Catlab.Graph, 5)
add_part!(h, :E, src=3, tgt=4)
add_part!(h, :V)
add_part!(h, :E, src=4, tgt=6)

# compute its cover in the subgraph topology
# covering = [Subobject(g, V=arr) for arr in Iterators.partition(parts(g, :V), 3)]

struct InducedGraph
    subobject::Subobject
    relabeling
end

see(g::InducedGraph) = see(g.subobject)

# modification of neighbors
# TODO is this truly the induced subgraph?
function n(g::Catlab.Graph, v; relabeling=identity)
    es_src = incident(g, v, :src)
    es_tgt = incident(g, v, :tgt)
    vs = setdiff(unique(g[es_src, :tgt] ∪ g[es_tgt, :src]), [v])
    subobject = ¬¬Subobject(g, V=vs, E=es_src ∪ es_tgt)
    InducedGraph(subobject, relabeling ∘ hom(subobject).components.V) 
end
n(g::Subobject, v; kwargs...) = n(dom(hom(g)), v; kwargs...)
n(g::InducedGraph, v) = n(g.subobject, v; relabeling = g.relabeling)

function m(g::Catlab.Graph, v; relabeling=identity)
    es_src = incident(g, v, :src)
    es_tgt = incident(g, v, :tgt)
    nes = setdiff(parts(g, :E), es_src ∪ es_tgt)
    nvs = setdiff(parts(g, :V), v ∪ g[es_src, :tgt] ∪ g[es_tgt, :src])
    subobject = ¬¬Subobject(g, V=nvs, E=nes)
    InducedGraph(subobject, relabeling ∘ hom(subobject).components.V)
end
m(g::Subobject, v; kwargs...) = m(dom(hom(g)), v; kwargs...)
m(g::InducedGraph, v) = m(g.subobject, v; relabeling = g.relabeling)

@data MDTree begin
    Leaf(v::Int)
    Parallel(v::Int, children::Vector{MDTree})
    Series(v, children::Vector{MDTree})
    Prime(v, children::Vector{MDTree})
end

function leaves(t::MDTree)
    out = []
    f = @λ begin
        (d, t::Leaf) => push!(out, (t.v, d))
        (d, t) => [f(([d..., idx], c)) for (idx, c) in enumerate(t.children)]
    end
    _ = f((0, t))
    return Dict(out)
end

Catlab.nv(s::Subobject) = nv(dom(hom(s)))
Catlab.nv(g::InducedGraph) = nv(g.subobject)

Catlab.parts(s::Subobject, kwargs...) = parts(dom(hom(s)), kwargs...)
Catlab.parts(g::InducedGraph, kwargs...) = parts(g.subobject, kwargs...)

function is_module(g::Catlab.Graph, M::Vector)
    outside = setdiff(parts(g, :V), M)
    state = []
    for u in outside
        nbrs = (inneighbors(g, u) ∪ outneighbors(g, u)) ∩ M |> collect
        isempty(nbrs) && continue
        if isempty(state)
            state = nbrs
        else
            state === nbrs || return false
        end
    end
    return true
end

function decomp(g)
    nv(g) == 0 && return nothing
	nv(g) == 1 && return (g isa InducedGraph ? Leaf(g.relabeling(1)) : Leaf(1))
    # TODO
    pivot = first(parts(g, :V))
    gl, gr = n(g, pivot), m(g, pivot)
    left, right = decomp(gl), decomp(gr)
    #
	pivot_original = g isa InducedGraph ? g.relabeling(pivot) : pivot
    pivot_leaf = Leaf(pivot_original)
    children = filter(!isnothing, [pivot_leaf, left, right])
    return Prime(pivot_original, children)
end

Catlab.vertices(i::InducedGraph) = i.relabeling.(parts(dom(hom(i.subobject)), :V))

Base.getindex(t::T, path...) where T <: MDTree = root(t, path)
function root(t::T, path::NTuple{N,Int}) where {N,T <: MDTree}
  if isempty(path) || t isa Leaf
	t
  else
	root(t.children[path[1]], path[2:end])
  end
end

function is_module(g::Catlab.Graph, M::Vector{Int}, witnesses::Vector{Int})::Bool
	M_set = Set(M)
    for w in witnesses
        # collect w's neighbors in G (in + out for directed, or just neighbors for undirected)
        w_out = g[incident(g, w, :src), :tgt]
        w_in  = g[incident(g, w, :tgt), :src]
        w_nbrs = Set(w_out) ∪ Set(w_in)
        adj_in_M = length(w_nbrs ∩ M_set)
        if adj_in_M != 0 && adj_in_M != length(M)
            return false   # w splits M
        end
    end
    return true
end

# restrict
function restrict(t::MDTree, g::Catlab.Graph, witnesses::Vector{Int})
	if is_module(g, collect(keys(leaves(t))), witnesses)
		return [t]
	else
		forest = []
		for child in t.children
			append!(forest, restrict(child, g, witnesses))
		end
		return forest
	end
	# which nonneighboring vertices neighbor the leaf
	leaftuples = [(leaf, witnesses .∈ Ref(neighbors(g, leaf))) for (leaf, _) in ls]
	# these are the adjacency matrices for nonneighboring vertices
	nbrs = Dict(filter(nbrs -> any(==(true), nbrs[2]), leaftuples))
end
restrict(t::MDTree, g::Catlab.Graph, nonneighbors::InducedGraph) = restrict(t, g, vertices(nonneighbors))

# ------

G = Catlab.Graph()
add_parts!(G, :V, 6)
# add edges (both directions if you want undirected behavior)
for (s,t) in [(1,2),(2,1),(1,3),(3,1),(2,3),(3,2),(4,5),(5,4),
              (6,1),(1,6),(6,2),(2,6),(6,3),(3,6)]
    add_part!(G, :E, src=s, tgt=t)
end

T = Parallel(0, [
    Series(0, MDTree[Leaf(1), Leaf(2), Leaf(3)]),
    Series(0, MDTree[Leaf(4), Leaf(5)])
])

F = restrict(T, G, [6])
@assert length(F) == 2
@assert Set(keys(leaves(F[1]))) == Set([1,2,3])
@assert Set(keys(leaves(F[2]))) == Set([4,5])


# just add the missing links to turn each collection (subgraph)
# into a simply embedding subgraph

# COROLLARY 2.1, for X module, rest: T(G) → T(G|X).
# for Y ⊆ V(G), the maximal modules of G in Y are a partition of Y.
# T(G) → Y (not T(Y)?) is the forest of trees obtained by restricting T(G) to the maximal modules of G contained in Y.

# A2

# struct ComplementStack
#     vec::Vector{Any}
#     lastModified::Time
# end

# function Base.push!(s::ComplementStack, x::Vector; complement=false)
#     remaining = setdiff(s.vec, x)
#     push!(s.vec, x)
# end

# Base.pop(s::ComplementStack) = s.vec[end]


# function is_simply_connected(s::Subobject, g::Graph)
# end

# ~(x,y) = is_connected(x,y)

# # cast neighbors as a Subobject

# cover, = covering # these are now matrices
# # for every column outside the graph cover
# for each v in graph \ cover
#     # if there is connection
#     if findfirst(~(v), cover)
#         # enbiggen the cover
#         union!(cover, v)
#     end
# end

# # A3


# # Decomposition algo

# # Graph --> Tree

