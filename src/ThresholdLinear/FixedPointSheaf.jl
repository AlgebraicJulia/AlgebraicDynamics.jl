
abstract type AbstractPresheaf end
# TODO

"""    The Fixed Point Support Functor is a separated presheaf. This is a section of that presheaf, which associates a neighborhood of a graph with its fixed point supports, if there are any. """
@struct_hash_equal mutable struct FPFunctor <: AbstractPresheaf
    base::Union{Graph, ImplicitGraph}
    data::Union{FPSections, Missing}
    # fixed point support. we may not want to compute the fixed point support when constructing the graph
end
export FPFunctor

# idempotence
FPFunctor(section::FPFunctor) = section

function Base.show(io::IO, fp::FPFunctor)
    print(io, """Fixed Point Supports
          G: V=$(nv(fp.base)), E=$(ne(fp.base))
          #sections: $(ismissing(fp.data) ? "Not Computed Yet" : length(fp.data))
          """)
end

# if consumes a ImplicitGraph, then load known information
function FPFunctor(g::Union{Graph, ImplicitGraph}; compute::Bool=true, hat::Bool=true)
    FPFunctor(g, compute ? FPSections(g) : missing)
end

nv(g::FPFunctor) = nv(g.base)
ne(g::FPFunctor) = ne(g.base)

# computes local supports
function FPFunctor(section::FPFunctor; params = DEFAULT_PARAMETERS, kwargs...)::FPFunctor
    section.data = if section.base isa Graph
        FPSections(section.base; params=params, kwargs...)
    else
        FPSections(Graph(section.base); params=params, kwargs...)
    end
    section
end

# idempotence
Graph(section::FPFunctor) = section

# Interactions

function Base.:+(G::Graph, H::ImplicitGraph)
    disjoint_union(G, Graph(H))
end

Base.:+(G::ImplicitGraph, H::Graph) = H + G

function Base.:+(G::FPFunctor, H::ImplicitGraph)
    G + FPFunctor(H)
end

Base.:+(H::ImplicitGraph, G::FPFunctor) = G + H

function Base.:*(G::FPFunctor, H::ImplicitGraph)
    G * FPFunctor(H)
end

Base.:*(H::ImplicitGraph, G::FPFunctor) = H * G

# FPFunctors are graphs with extra data
# function Base.:+(G::FPFunctor, H::Union{Graph, ImplicitGraph})
#     H = FPSections(FPFunctor(H))
#     X = disjoint_union(G.base, H.base)
#     H_data = shift(H
# end

# # commutativity
# Base.:+(G::Union{Graph, ImplicitGraph}, H::FPFunctor) = H + G

function shift(g::FPFunctor, offset::Int)
    graph = shift(g.base, offset)
    data = shift(g.data, offset)
    FPFunctor(graph, data)
end

# TODO
shift(g::Graph, args...) = g

# what happens if the "base" is sortable? (impl PartialEq) 
function Base.sort!(g::FPFunctor)
    sort!(g.data)
end

function Base.:+(G::FPFunctor, H::FPFunctor)
    H_shifted = shift(H, nv(G.base))
    X = disjoint_union(G.base, H_shifted.base)
    # TODO ImplicitGraphs need to store their offset
    FPFunctor(X, disjoint_union(G.data, H_shifted.data))
end

# TODO
function Base.:*(G::FPFunctor, H::FPFunctor)
    H_shifted = shift(H, nv(G.base))
    X = clique_union(G.base, H_shifted.base)
    # TODO ImplicitGraphs need to store their offset
    FPFunctor(X, clique_union(G.data, H_shifted.data))
end

# @data ExpressionTree begin
#     BruteForce(::FPFunctor) # may or may not have data computed already
#     CliqueUnion(::Vector{ExpressionTree})
#     DisjointUnion(::Vector{ExpressionTree})
#     CycleUnion(::Vector{ExpressionTree})
# end

# ExpressionTree
@data GluingRule <: GraphArchitecture begin
    Terminal(::FPFunctor) # Brute Force
    CliqueUnion(::Vector{<:GraphArchitecture})
    DisjointUnion(::Vector{<:GraphArchitecture})
    CycleUnion(::Vector{<:GraphArchitecture})
end
export GluingRule, Terminal, DisjointUnion, CliqueUnion, CycleUnion

# convenience methods
GluingRule(t::GluingRule) = t
GluingRule(t::FPFunctor) = Terminal(t)
GluingRule(g::Union{Graph, ImplicitGraph}) = Terminal(FPFunctor(g))
GluingRule(g::Vector{<:GluingRule}) = DisjointUnion(g)
GluingRule(k::Int) = Terminal(FPFunctor(Graph(k))) # TODO Implicit

# function Base.show(io::IO, d::DisjointUnion)
    # printio

CliqueUnion(args...) = CliqueUnion([GluingRule.(args)...])

# a variadic call to DisjointUnion
function DisjointUnion(args...)
    DisjointUnion([GluingRule.(args)...])
end

# A+B = (B+A)'
# A*B = (B*A)'
# A+(B*C) = 2((A+B)*C*A) - (A+B+(C+A))
Base.:*(G::GluingRule, H::GluingRule) = CliqueUnion(G, H)
Base.:+(G::GluingRule, H::GluingRule) = DisjointUnion([G, H])

function Base.:+(G::DisjointUnion, H::DisjointUnion)
    DisjointUnion(G._1..., H._1...)
end

nv(t::CliqueUnion) = foldl(+, nv.(t._1))
nv(t::DisjointUnion) = foldl(+, nv.(t._1))

# TODO didn't want to write the foldl
ne(t::CliqueUnion) = ne(Graph(t))
ne(t::DisjointUnion) = ne(Graph(t))

Graph(t::Terminal) = Graph(t._1)

""" Renders a disjoint union of graphs into a single graph """
Graph(G::DisjointUnion) = foldl(+, Graph.(G._1))

# TODO need to amend for FpSupport graph
function Graph(g::CliqueUnion)
    X = Graph()
    foldl(*, Graph.(g._1), init=X)
end

FPFunctor(g::GluingRule) = @match g begin
    Terminal(g) => ismissing(g.data) ? g : g
    DisjointUnion(g) => foldl(+, FPFunctor.(g))
    CliqueUnion(g) => foldl(*, FPFunctor.(g))
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
