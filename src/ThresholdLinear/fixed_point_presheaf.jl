# TODO connect to Catlab Sheaf module
abstract type AbstractPresheaf end

"""    The Fixed Point Support Functor is a separated presheaf. This is a section of that presheaf, which associates a neighborhood of a graph with its fixed point supports, if there are any. """
@struct_hash_equal mutable struct FP <: AbstractPresheaf
    base::Union{Graph, ImplicitGraph}
    data::Union{FPSections, Missing}
    # fixed point support. we may not want to compute the fixed point support when constructing the graph
end
export FP

function Base.show(io::IO, fp::FP)
    print(io, """Fixed Point Supports
          G: V=$(nv(fp.base)), E=$(ne(fp.base))
          #sections: $(ismissing(fp.data) ? "Not Computed Yet" : length(fp.data))
          """)
end

# if consumes a ImplicitGraph, then load known information
function FP(g::Union{Graph, ImplicitGraph}; compute::Bool=true)
    FP(g, compute ? FPSections(g) : missing)
end

nv(g::FP) = nv(g.base)
ne(g::FP) = ne(g.base)

# computes local supports if the data is missing.
function FP(fp::FP; params = DEFAULT_PARAMETERS, kwargs...)::FP
    ismissing(fp.data) || return fp 
    fp.data = FPSections(fp.base isa Graph ? section.base : Graph(section.base); params=params, kwargs...)
    fp
end

# Interactions

function Base.:+(G::Graph, H::ImplicitGraph)
    disjoint_union(G, Graph(H))
end

Base.:+(G::ImplicitGraph, H::Graph) = H + G

function Base.:+(G::FP, H::ImplicitGraph)
    G + FP(H)
end

Base.:+(H::ImplicitGraph, G::FP) = G + H

function Base.:*(G::FP, H::ImplicitGraph)
    G * FP(H)
end

Base.:*(H::ImplicitGraph, G::FP) = H * G

function shift(g::FP, offset::Int)
    graph = shift(g.base, offset)
    data = shift(g.data, offset)
    FP(graph, data)
end

# TODO we shift to avoid colliding vertex labels. Plain graphs do not have labels, as they do not have attributes at all, so shifting it does nothing at all.
shift(g::Graph, args...) = g

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

function Base.:+(fp::FP, g::GluingExpression)
    DisjointUnion(Terminal(fp), g)
end
# commutativity
Base.:+(g::GluingExpression, fp::FP) = fp + g

Base.:+(G::GluingExpression, H::GluingExpression) = DisjointUnion(G, H)

nv(t::DisjointUnion) = foldl(+, nv.(t._1))
# TODO didn't want to write the foldl
ne(t::DisjointUnion) = ne(Graph(t))

""" Renders a disjoint union of graphs into a single graph """
Graph(G::DisjointUnion) = foldl(+, Graph.(G._1))

# #############
# CLIQUE UNION
# #############

CliqueUnion(args...) = CliqueUnion([GluingExpression.(args)...])

function Base.:*(fp1::FP, fp2::FP)
    CliqueUnion(Terminal(fp1), Terminal(fp2))
end

function Base.:*(fp::FP, g::GluingExpression)
    CliqueUnion(Terminal(fp), g)
end

Base.:*(g::GluingExpression, fp::FP) = fp * g

Base.:*(G::GluingExpression, H::GluingExpression) = CliqueUnion(G, H)

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

function (↻)(fp1::FP, fp2::FP)
    CyclicUnion(Terminal(fp1), Terminal(fp2))
end

function (↻)(fp::FP, g::GluingExpression)
    CyclicUnion(Terminal(fp), g)
end

(↻)(g::GluingExpression, fp::FP) = fp ↻ g

(↻)(G::GluingExpression, H::GluingExpression) = CyclicUnion(G, H)
export ↻

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

"""    indicator(g::AbstractGraph, σ::Vector{Int})

Get the indicator function of a subset with respect to a graph. Returns a vector of 0 and 1s.

This should probably be a FinFunction.
"""
function indicator(g::AbstractGraph, σ::Vector{Int})
    [Int(v ∈ σ) for v ∈ vertices(g)]
end
export indicator
