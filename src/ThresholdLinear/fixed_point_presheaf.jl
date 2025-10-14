# TODO connect to Catlab Sheaf module
abstract type AbstractPresheaf end

"""    The Fixed Point Support Functor is a separated presheaf. This is a section of that presheaf, which associates a neighborhood of a graph with its fixed point supports, if there are any. """
@struct_hash_equal mutable struct FP <: AbstractPresheaf
    base::Union{Graph, ImplicitGraph}
    data::Union{FPSections, Missing} 
end
export FP
# Typing `data` as also `::Missing` is perhaps overengineering, but it is conceivable that we may not want to compute the fixed point support when constructing the graph. Recall that in this framework, a graph is glued together by gluing rules, which is abstractly represented as an abstract syntax tree, and that terms of the tree whose fixed-point supports are computed with brute force are called "terminals." While decompositional methods might be able to simplify a terminal even further, its possible that a large terminal graph can take a long time to compute in brute force. For example, a Cyclic Graph for example takes a long time to compute the fixed points. While we overcome that specific problem with an ImplicitGraph, there could be other cases. In essence, allowed for "missing" information helps us separate (1) building the expression tree for a graph from (2) its evaluation, where all fixed point supports must be known.
# TODO give a meaningful example

function Base.show(io::IO, fp::FP)
    print(io, """Fixed Point Supports
          G: V=$(nv(fp.base)), E=$(ne(fp.base))
          #sections: $(ismissing(fp.data) ? "Not Computed Yet" : length(fp.data))
          """)
end

# if consumes a ImplicitGraph, then load known information. The docs at this point do not demonstrate "missing" information, as fixed point supports are automatically computed
function FP(g::Union{Graph, ImplicitGraph}; compute::Bool=true)
    FP(g, compute ? FPSections(g) : missing)
end

# two basic properties of graphs (number of vertices and edges) are passed through here.
nv(g::FP) = nv(g.base)
ne(g::FP) = ne(g.base)

# computes local supports if the data is missing.
function FP(fp::FP; params = DEFAULT_PARAMETERS, kwargs...)::FP
    ismissing(fp.data) || return fp
    base = fp.base isa Graph ? section.base : Graph(section.base)
    fp.data = FPSections(base; params=params, kwargs...)
    fp
end

# Interactions
# TODO this boilerplate can be removed if the lowercase "disjoint_", "clique_", and "cyclic_", union methods were fetched from the Disjoint_, Clique_, Cyclic_ terms in the MLStyle ADT.
# ```julia
# method(::T, args...) = T_union(args...) 
# ```

function shift(g::FP, offset::Int)
    graph = shift(g.base, offset)
    data = shift(g.data, offset)
    FP(graph, data)
end

# what happens if the "base" is sortable? (impl PartialEq) 
function Base.sort!(g::FP)
    sort!(g.data)
end

function disjoint_union(G::FP, H::FP)
    H_shifted = shift(H, nv(G.base))
    X = disjoint_union(G.base, H_shifted.base)
    FP(X, disjoint_union(G.data, H_shifted.data))
end

function clique_union(G::FP, H::FP)
    H_shifted = shift(H, nv(G.base))
    X = clique_union(G.base, H_shifted.base)
    FP(X, clique_union(G.data, H_shifted.data))
end

function cyclic_union(G::FP, H::FP)
    H_shifted = shift(H, nv(G.base))
    X = cyclic_union(G.base, H_shifted.base)
    FP(X, cyclic_union(G.data, H_shifted.data))
end

@data GluingExpression begin
    Terminal(::FP) # Brute Force
    CliqueUnion(::Vector{<:GluingExpression})
    DisjointUnion(::Vector{<:GluingExpression})
    CyclicUnion(::Vector{<:GluingExpression})
end
export GluingExpression, DisjointUnion, CliqueUnion, CyclicUnion

# convenience methods
GluingExpression(t::GluingExpression) = t
GluingExpression(t::FP) = Terminal(t)
GluingExpression(g::Union{Graph, ImplicitGraph}) = Terminal(FP(g))
GluingExpression(g::Vector{<:GluingExpression}) = DisjointUnion(g)
GluingExpression(k::Int) = Terminal(FP(DiscreteGraph(k))) 

# ##############
# DISJOINT UNION
# ##############

# a variadic call to DisjointUnion
function DisjointUnion(args...)
    DisjointUnion([GluingExpression.(args)...]) # wraps graphs 
end

function Base.:+(fp1::FP, fp2::FP)
    DisjointUnion(Terminal(fp1), Terminal(fp2))
end

# the disjoint union between a fixed point functor over a graph and an ordinary implicit graph will first apply the FP functor to the implicit graph before computing the disjoint union.
Base.:+(G::FP, H::ImplicitGraph) = G + FP(H)
Base.:+(H::ImplicitGraph, G::FP) = G + H

Base.:+(fp::FP, g::GluingExpression) = DisjointUnion(Terminal(fp), g)
Base.:+(g::GluingExpression, fp::FP) = fp + g
Base.:+(G::GluingExpression, H::GluingExpression) = DisjointUnion(G, H)

nv(t::DisjointUnion) = foldl(+, nv.(t._1))
# TODO didn't want to write the foldl
ne(t::DisjointUnion) = ne(Graph(t))

""" Renders a disjoint union of graphs into a single graph """
Graph(G::DisjointUnion) = foldl(+, Graph.(G._1))

# ############
# CLIQUE UNION
# ############

CliqueUnion(args...) = CliqueUnion([GluingExpression.(args)...])

Base.:*(fp1::FP, fp2::FP) = CliqueUnion(Terminal(fp1), Terminal(fp2))

Base.:*(fp::FP, g::GluingExpression) = CliqueUnion(Terminal(fp), g)

Base.:*(g::GluingExpression, fp::FP) = fp * g
Base.:*(G::GluingExpression, H::GluingExpression) = CliqueUnion(G, H)

Base.:*(G::FP, H::ImplicitGraph) = G * FP(H)
Base.:*(H::ImplicitGraph, G::FP) = H * G

nv(t::CliqueUnion) = foldl(+, nv.(t._1))
ne(t::CliqueUnion) = ne(Graph(t))

# TODO need to amend for FpSupport graph
function Graph(g::CliqueUnion)
    X = Graph()
    foldl(*, Graph.(g._1), init=X)
end

# #############
# CYCLIC UNION
# #############

CyclicUnion(args...) = CyclicUnion([GluingExpression.(args)...])

export ↻

(↻)(fp1::FP, fp2::FP) = CyclicUnion(Terminal(fp1), Terminal(fp2))
(↻)(fp::FP, g::GluingExpression) = CyclicUnion(Terminal(fp), g)

(↻)(G::FP, H::ImplicitGraph) = G * FP(H)
(↻)(H::ImplicitGraph, G::FP) = H * G

(↻)(g::GluingExpression, fp::FP) = fp ↻ g

(↻)(G::GluingExpression, H::GluingExpression) = CyclicUnion(G, H)

Graph(t::Terminal) = Graph(t._1)

# #################
# Gluing Expression
# #################

FP(g::GluingExpression) = @match g begin
    Terminal(g) => ismissing(g.data) ? FP(g) : g
    DisjointUnion(g) => foldl(disjoint_union, FP.(g))
    CliqueUnion(g) => foldl(clique_union, FP.(g))
    CyclicUnion(g) => foldl(cyclic_union, FP.(g))
    err => error("$err")
end
