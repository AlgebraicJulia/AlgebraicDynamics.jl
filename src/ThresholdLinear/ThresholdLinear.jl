module ThresholdLinear
using LinearAlgebra
using SparseArrays
using Catlab
using Catlab.Graphs
using Catlab.Graphics.Graphviz: view_graphviz
using Combinatorics
import SciMLBase: ODEProblem, NonlinearProblem, ODESolution, solve

export dynamics, nldynamics 

# Contains utilities for creating and combining graphs.
include("graph_utils.jl")

# Contains code for defining fixed point supports, elements of the fixed point presheaf. `Support` is a wrapper for vectors of indices or iterators which represent vectors of indices. It also produces the indices of vectors of non-integer numbers which are greater than a certain (small) quantity. 
include("supports.jl")

# Contains code for creating and combining fixed point presheafs localized over covers.
include("fixed_point_presheaf.jl")

draw(g) = to_graphviz(g, node_labels=true)
export draw

see(g) = view_graphviz(draw(g))
export see

"""    LegalParameters{F}

The parameters of a CTLN model are:
  - epsilon
  - delta
  - theta

The constructor enforces the constraint that ϵ < δ/(δ+1).
"""
struct LegalParameters{F}
  epsilon::F
  delta::F
  theta::F

  function LegalParameters{F}(ϵ::F, δ::F, θ::F) where F
    @assert ϵ > 0
    @assert δ > 0
    @assert θ > 0
    @assert ϵ < δ / (δ + 1)
    new(ϵ, δ, θ)
  end
end
export LegalParameters

"""    DEFAULT_PARAMETERS = (ϵ=0.25, δ=0.5, θ=1.0)
"""
const DEFAULT_PARAMETERS = LegalParameters{Float64}(0.25, 0.5, 1.0)
export DEFAULT_PARAMETERS

"""    CTLNetwork{F}

  - G::Graph
  - parameters::LegalParameters{F}

Stores a combinatorial linear threshold network as a graph and the parameters.
The single argument constructor uses the DEFAULT_PARAMETERS.
"""
struct CTLNetwork{F}
    G::Graph
    parameters::LegalParameters{F}
end
export CTLNetwork

CTLNetwork(G::Graph) = CTLNetwork(G, DEFAULT_PARAMETERS)

# passthrough method
Catlab.nv(ctln::CTLNetwork) = nv(ctln.G)

function adjacency_matrix(g::AbstractGraph)::SparseMatrixCSC{Float64, Int64}
    n = nv(g)
    J = g[:, :src]
    I = g[:, :tgt]
    V = ones(nparts(g, :E))
    sparse(I, J, V, n, n)
end
export adjacency_matrix

adjacency_matrix(ctln::CTLNetwork) = adjacency_matrix(ctln.G)

"""     TLNetwork{T}

  - W::AbstractMatrix{T}
  - b::AbstractVector{T}

Stores a Threshold Linear Network as a Matrix of weights and Vector of biases.
You can construct these from a Graph with [`CTLNetwork`](@ref).
"""
struct TLNetwork{T}
    W::AbstractMatrix{T}
    b::AbstractVector{T}
end
export TLNetwork

Base.size(tln::TLNetwork) = size(tln.W)

"""    TLNetwork(g::CTLNetwork)

Convert a CTLN to a TLN by realizing the weights an biases for that Graph.
"""
function TLNetwork(ctln::CTLNetwork)
    n = nv(ctln)
    p = ctln.parameters
    W = (I - ones(n, n)) .* (1 + p.delta)
    W += adjacency_matrix(ctln.G) * (p.epsilon + p.delta)
    b = ones(n) .* p.theta
    TLNetwork(W, b)
end

"""    relu(x::Number)

Nonlinear function used in nonlinear dynamics
"""
relu(x::Number) = maximum([x, 0])

"""    dynamics(tln::TLNetwork)

Construct the vector field for the TLN. You can pass this to an ODEProblem, or as a Continuous Resource Sharer.
"""
function dynamics(tln::TLNetwork)
    f(u) = begin
      relu.(tln.W * u .+ tln.b) .- u
    end
    return f
end

# Dynamics is a callable struct. What if we just passed this instead?
# function (tln::TLNetwork)(u)
#     relu.(tln.W * u .+ tln.b) .- u
# end

"""    nldynamics(tln::TLNetwork)

Construct the root finding problem for the TLN. You can pass this to an NonlinearProblem to find the steady states of the network.
"""
function nldynamics(tln::TLNetwork)
    f(u, p) = relu.(tln.W * u .+ tln.b) .- u
    return f
end

function dynamics(du, u, p::TLNetwork, t)
    mul!(du, p.W, u)
    du .= relu.(du .+ p.b) .- u
end

ODEProblem(tln::TLNetwork, u₀, tspan) = ODEProblem(dynamics, u₀, tspan, tln)
ODEProblem(tln::CTLNetwork, u₀, tspan) = ODEProblem(TLNetwork(tln), u₀, tspan)

function NonlinearProblem(tln::TLNetwork, u₀)
    fₚ = nldynamics(tln)
    NonlinearProblem(fₚ, u₀, tln)
end

function NonlinearProblem(tln::CTLNetwork, u₀)
    NonlinearProblem(TLNetwork(tln), u₀)
end

struct Section
    support::AbstractVector{Int} # Support?
    section::Vector{Float64}
end
export Section

uniform(n::Int) = ones(n) ./ n
uniform(g::AbstractGraph) = uniform(nv(g))

"""    restriction_fixed_point(G::AbstractGraph, V::AbstractVector{Int}, parameters=DEFAULT_PARAMETERS)

Restrict a graph to the induced subgraph given by `V` and then solve for its fixed point support.

Returns the fixed point of G corresponding to nonzeros in V padded with zeros for the vertices that aren't in V.
"""
function restriction_fixed_point(G::AbstractGraph, V::AbstractVector{Int}, parameters=DEFAULT_PARAMETERS)
    g = induced_subgraph(G, V)
    tln = TLNetwork(CTLNetwork(g, parameters))
    prob = NonlinearProblem(tln, uniform(g))
    fp = solve(prob)
    # this returns the indices of an ODESolution which are effectively nonzero.
    # `σ` is the vector of vertices which constitute the support. 
    support_indices = Support(fp)
    support = V[support_indices]
    # ... 
    section = zeros(nv(G))
    map(support_indices) do v
        section[V[v]] = fp.u[v]
    end
    return Section(support, section)
end
export restriction_fixed_point

# Helper functions for the sheaf algorithm to compute fixed point supports of a CTLN.

# TODO: Check that this is correct. Because we should be getting the same results for
# clique_union and disjoint_union covers.
# Also write some unit tests for this.
"""   is_fp_support(net::CTLNetwork, nodes::Vector{Int}) -> Bool   

Checks if the given subset of nodes constitutes a fixed point support of the given TLN (see Alg. 1 in paper).
"""
function is_fp_support(net::TLNetwork, nodes::Vector{Int})::Bool
    W = net.W
    b = net.b
    n, = size(W)
    W_r = W[nodes, nodes]
    b_r = b[nodes]
    # computes the fixed point at a certain pt
    x_star_r = (I - W_r) \ b_r
    x_star = zeros(n)
    x_star[nodes] = x_star_r
    #
    on_neuron_condition = true
    off_neuron_condition = true
    y = W * x_star + b
    for i in 1:n
        y_i = y[i] 
        if i in nodes
            on_neuron_condition = on_neuron_condition && y_i > 0
        else
            off_neuron_condition = off_neuron_condition && y_i <= 0 
        end
    end
    return on_neuron_condition && off_neuron_condition
end
export is_fp_support

function is_fp_support(net::TLNetwork, support::Support)
    is_fp_support(net, support.indices)
end

# FP(g::ImplicitGraph) = FP(g, FPSections(g))


# TODO fix signature
"""   FPSections(net::TLNetworks)::FPSections()

Uses a brute force search to enumerate the set of fixed point supports of a given CTLN.
"""
function FPSections(net::TLNetwork)
    W = net.W
    n, = size(net)
    supports = FPSections()
    for support in powerset(1:n)
        if is_fp_support(net, support)
            push!(supports, Support(support))
        end
    end
    supports
end
export FPSections

# TODO used to be "global_from_local." It's called "Supports" because it computes a support using information..
function FPSections(ctln::CTLNetwork, locals::Vector{FPSections})
    locals = deepcopy(locals)
    # add empty supports to each local
    point!.(locals)
    # instantiate the Threshold Linear Network
    tln = TLNetwork(ctln)
    # Set of computed supports
    computed = Set{Support}()
    # valid global supports
    out = FPSections()
    # cartesian product from local supports
    for tuple_of_supports in product(locals)
        support = union(tuple_of_supports...)
        allequal([Int[], support.indices...]) && continue
        unique!(support)
        sort!(support)
        if support ∉ computed
            if is_fp_support(tln, support)
                push!(out, support)
            end
            push!(computed, support)
        end
    end
    return out
end

FPSections(ctln::CTLNetwork, locals::Vector{Support}) = FPSections(ctln, FPSections.(locals))

"""   simply_embedded_cover(net::CTLNetwork)   

computes a simply embedded cover (modular decomposition) of a given CTLN.
"""
function simply_embedded_cover(net::CTLNetwork)
end

function enumerate_supports(net::CTLNetwork, cover)::Vector{Vector{Int}}
end

end # module
