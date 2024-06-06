module ThresholdLinear
using LinearAlgebra
using SparseArrays
using Catlab
using Catlab.Graphs
import SciMLBase: ODEProblem, NonlinearProblem, ODESolution

export draw, support, add_reflexives!, add_reflexives, adjacency_matrix,
  LegalParameters, DEFAULT_PARAMETERS, CTLNetwork, TLNetwork, dynamics, nldynamics

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

const DEFAULT_PARAMETERS = LegalParameters{Float64}(0.25, 0.5, 1.0)

struct CTLNetwork{F}
  G::Graph
  parameters::LegalParameters{F}
end

CTLNetwork(G::Graph) = CTLNetwork(G, DEFAULT_PARAMETERS)

struct TLNetwork{T}
  W::AbstractMatrix{T}
  b::AbstractVector{T}
end

function TLNetwork(g::CTLNetwork)
  n = nv(g.G)
  p = g.parameters
  W = (I - ones(n,n)) .* (1+p.delta)
  W += adjacency_matrix(g.G) * (p.epsilon + p.delta)
  b = ones(n) .* p.theta
  TLNetwork(W,b)
end

relu(x) = x > 0 ? x : 0 

function dynamics(tln::TLNetwork)
  f(u) = begin
    relu.(tln.W*u .+ tln.b) .- u
  end
  return f
end

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
end