using Base.Iterators: product, ProductIterator, Flatten
using StructEquality
using Catlab.Graphs
using Catlab
using Catlab.CategoricalAlgebra
using AlgebraicDynamics
using AlgebraicDynamics.ThresholdLinear

"""    Support

In the sense of CTLNs, a support is a subset of vertices where dynamics enjoys a fixed point.

supp(x*) := { i | x_i^* > 0 }

These points are contained by this struct, however we do not enforce the support condition unless we are ingesting a vector of floats. 
"""
@struct_hash_equal struct Support
    indices::Union{Flatten, Vector{Int}}
   
    # empty support
    Support() = new(Int[])
    Support(indices::Vector{Int}) = new(indices)
    # TODO what if the iterate were not Int?
    Support(itr::Flatten) = new(itr)

    # The support underlying a vector of floating points is the vector of indices
    # which obey the support condition
    function Support(x::AbstractVector, ϵ::Real=1e-12)
        new([i for i in 1:length(x) if abs(x[i]) > ϵ])
    end
    
    # the support underlying an ODE solution is a collection of indices of the floating points
    # which are above a certain threshold.
    function Support(soln::ODESolution, ϵ::Real=1e-12)
        new(soln.u[end], ϵ)
    end
end
export Support

function Base.show(io::IO, support::Support)
    if !isempty(support.indices)
        print(io, "σ$(support.indices)")
    else
        print(io, "σ[]")
    end
end

Base.unique!(support::Support) = Support(unique!(support.indices))
Base.isempty(support::Support) = isempty(support.indices)

function Base.union(supports::Vararg{Support, N}) where N
    Support(reduce(vcat, getfield.(supports, :indices)))
end

Base.sort!(support::Support) = Support(sort!(support.indices))

function Base.getindex(v::AbstractVector{Int}, support::Support)
    v[support.indices]
end

function shift(support::Support, n::Int)
    Support([idx + n for idx in support.indices])
end
export shift

function Base.isless(s::Support, t::Support)
    s.indices < t.indices
end

"""    FPSections 
Struct containing local supports on a graph. Since this is a covering, every vertex is included.
"""
@struct_hash_equal struct FPSections
    supports::Vector{Support}

    FPSections() = new(Support[])
    FPSections(support::Support) = new([support])
    FPSections(supports::Vector{Support}) = new(supports)
    FPSections(ns::Vector{Int}) = new([Support(ns)])
end
export FPSections

Base.push!(l::FPSections, support::Support) = push!(l.supports, support)
Base.push!(l::FPSections, x::Vector{Int}) = push!(l.supports, Support(x))

function Base.union!(l::FPSections, supports::Vector{Support})
    FPSections(union!(l.supports, supports))
end
Base.union!(l::FPSections, support::Support) = union!(l, [support])

# TODO need method `isless` for Support
Base.sort(l::FPSections) = FPSections(sort(l.supports))

Base.length(fp::FPSections) = length(fp.supports)

# Base.eachindex(locals::FPSectionss) = eachindex(locals.data)
# Base.getindex(locals::FPSectionss, key...) = getindex(locals.data, key...)

""" This adds a "point" to the set, a distinguished empty support
"""
point!(l::FPSections) = union!(l, Support())
export point!

point(l::FPSections) = union(l, Support())
export point

shift(l::FPSections, n::Int) = FPSections(shift.(l.supports, n))

function Iterators.product(l::Vector{FPSections})
    Iterators.product(getfield.(l, :supports)...)
end

function Base.maximum(support::Support)
    maximum(support.indices)
end

function Base.maximum(l::FPSections)
    maximum(maximum.(l.supports))
end

function disjoint_union(l::FPSections, m::FPSections; perform_shift::Bool=false)
    if perform_shift
        m = shift(m, maximum(l))
    end
    # TODO only valid for Disjoint Union. Clique union cannot union those guys
    FPSections(l.supports ∪ m.supports ∪ collect(product([l, m])))
end

function clique_union(l::FPSections, m::FPSections; perform_shift::Bool=false)
    if perform_shift
        m = shift(m, maximum(l))
    end
    # TODO only valid for Disjoint Union. Clique union cannot union those guys
    FPSections(collect(product([l, m])))
end

function Catlab.:∨(l::FPSections, m::FPSections)
    disjoint_union(l, m; perform_shift=true)
end
export ∨
# vee

function Base.union(s::Support, t::Support)
    Support(union(s.indices, t.indices))
end

function Base.union(ℓ::FPSections, m::FPSections)
    FPSections(union(ℓ.supports..., m.supports...))
end

function Base.union(ℓ::FPSections, s::Support)
    FPSections([ℓ.supports..., s])
end

function Base.union(ℓ::FPSections, ℓs...)
    @assert eltype(ℓs) == FPSections
    FPSections(ℓ.supports..., getfield.(ℓs, :supports)...)
end

function Base.collect(itr::ProductIterator{Tuple{Vector{Support}, Vector{Support}}})
    out = map(itr) do (s, t)
        s ∪ t
    end
    out[:] # XXX The supports become a matrix otherwise. Need to flatmap
end

# Formerly "compute_local_support"
# Return the fixed-point supports of a local in local indices 1:nv(locals).
function FPSections(graph::Graph; params = DEFAULT_PARAMETERS)::FPSections
    tln = TLNetwork(CTLNetwork(graph, params))
    FPSections(tln) # enumerate_support_tln
end

# from partition gives each block its nodes 
function cover_partition(partition::Vector{Int})
    # ex partition = [1,1,1,2,2,2,3,3,3]  (if 3 blocks)
    k = maximum(partition)  # ex k = 3
    n = length(partition)   # ex n = 9
    cover = [Int[] for _ in 1:k]    # ex cover = [[],[],[]] (if k=3) 
    # assign node to its block
    for v in 1:n
        push!(cover[partition[v]], v)   # ex cover = [[1,2,3],[4,5,6],[7,8,9]]
    end
    return cover 
end
export cover_partition

# @test cover_partition([1,1,1,2,2,2,3,3,3]) == [[1,2,3],[4,5,6],[7,8,9]]

"""    local_sup
This maps each local support to global indices, τ is the number of the nodes in the block
# e.g. τ = [1,2,3] for the first block, τ = [4,5,6] for the second block, etc.
"""
function local_sup(supports::Vector{Support}, τ::Vector{Int})::Vector{Support}
    ls = Support.([sort(τ[support.indices]) for support in supports])
    return ls
end
export local_sup
