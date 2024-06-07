module ThresholdLinear
using LinearAlgebra
using SparseArrays
using Catlab
using Catlab.Graphs
import SciMLBase: ODEProblem, NonlinearProblem, ODESolution, solve

export draw, support, add_reflexives!, add_reflexives, adjacency_matrix,
  LegalParameters, DEFAULT_PARAMETERS, CTLNetwork, TLNetwork, dynamics, nldynamics,
  indicator, restriction_fixed_point

draw(g) = to_graphviz(g, node_labels=true)

# support(x::AbstractVector) = [i for i in 1:length(x) if x[i] != 0.0] 
support(x::AbstractVector, eps::Real=1e-12) = [i for i in 1:length(x) if abs(x[i]) > eps] 
support(soln::ODESolution, eps::Real=1e-12) = support(soln.u[end], eps)

function add_reflexives!(g::AbstractGraph)
  for v in vertices(g)
    add_edge!(g, v,v)
  end
  return g
end
add_reflexives(g) = add_reflexives!(copy(g))

function adjacency_matrix(g::AbstractGraph)
  n = nv(g)
  J = g[:, :src]
  I = g[:, :tgt]
  V = ones(nparts(g, :E))
  sparse(I,J,V, n, n)
end

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
    @assert ϵ < δ/(δ+1)
    new(ϵ, δ, θ)
  end
end

"""    DEFAULT_PARAMETERS = (ϵ=0.25, δ=0.5, θ=1.0)
"""
const DEFAULT_PARAMETERS = LegalParameters{Float64}(0.25, 0.5, 1.0)

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

CTLNetwork(G::Graph) = CTLNetwork(G, DEFAULT_PARAMETERS)

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

"""    TLNetwork(g::CTLNetwork)

Convert a CTLN to a TLN by realizing the weights an biases for that Graph.
"""
function TLNetwork(g::CTLNetwork)
  n = nv(g.G)
  p = g.parameters
  W = (I - ones(n,n)) .* (1+p.delta)
  W += adjacency_matrix(g.G) * (p.epsilon + p.delta)
  b = ones(n) .* p.theta
  TLNetwork(W,b)
end

relu(x) = x > 0 ? x : 0 

"""    dynamics(tln::TLNetwork)

Construct the vector field for the TLN. You can pass this to an ODEProblem, or as a Continuous Resource Sharer.
"""
function dynamics(tln::TLNetwork)
  f(u) = begin
    relu.(tln.W*u .+ tln.b) .- u
  end
  return f
end

"""    nldynamics(tln::TLNetwork)

Construct the root finding problem for the TLN. You can pass this to an NonlinearProblem to find the steady states of the network.
"""
function nldynamics(tln::TLNetwork)
  f(u,p) = relu.(tln.W*u .+ tln.b) .- u
  return f
end

dynamics(du, u, p::TLNetwork,t) = begin
    mul!(du, p.W, u)
    du .= relu.(du .+ p.b) .- u
end

ODEProblem(tln::TLNetwork, u₀, tspan) = ODEProblem(dynamics, u₀, tspan, tln)
ODEProblem(tln::CTLNetwork, u₀, tspan) = ODEProblem(TLNetwork(tln), u₀, tspan)

NonlinearProblem(tln::TLNetwork, u₀) = begin
  fₚ = nldynamics(tln)
  NonlinearProblem(fₚ, u₀, tln)
end

NonlinearProblem(tln::CTLNetwork, u₀) = NonlinearProblem(TLNetwork(tln), u₀)

"""    indicator(g::AbstractGraph, σ::Vector{Int})

Get the indicator function of a subset with respect to a graph.
This should probably be a FinFunction.
"""
indicator(g::AbstractGraph, σ::Vector{Int}) = map(vertices(g)) do v
  if v in σ
    return 1
  else
    return 0
  end
end

"""    restriction_fixed_point(G::AbstractGraph, V::AbstractVector{Int}, parameters=DEFAULT_PARAMETERS)

Restrict a graph to the induced subgraph given by `V` and then solve for its fixed point support.

Returns the fixed point of G corresponding to nonzeros in V padded with zeros for the vertices that aren't in V.
"""
function restriction_fixed_point(G::AbstractGraph, V::AbstractVector{Int}, parameters=DEFAULT_PARAMETERS)
  g = induced_subgraph(G, V)
  tln = TLNetwork(CTLNetwork(g, parameters))
  prob = NonlinearProblem(tln, ones(nv(g)) ./ nv(g))
  fp = solve(prob)
  σg = support(fp)
  σ  = V[σg]
  u = zeros(nv(G))
  map(σg) do v
    u[V[v]] = fp.u[v]
  end
  return σ, u
end

end # module