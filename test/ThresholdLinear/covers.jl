using Catlab.Graphs
using Catlab.CategoricalAlgebra
using AlgebraicDynamics
using AlgebraicDynamics.ThresholdLinear
using Test

C3 = cycle_graph(Graph, 3)
D3 = discrete_graph = Graph(3)

function disjoint_union(G,H)
  X = Graph(nv(G) + nv(H))
  for e in edges(G)
    add_edge!(X, src(G, e), tgt(G, e))
  end
  for e in edges(H)
    add_edge!(X, src(H, e) + nv(G), tgt(H, e) + nv(G))
  end
  return X
end

function clique_union(G,H)
  X = disjoint_union(G,H)
  for v in vertices(G)
    for w in vertices(H)
      add_edge!(X, v, w + nv(G))
      add_edge!(X, w + nv(G), v)
    end
  end
  return X
end

G1 = disjoint_union(C3, D3)

@test ne(G1) == 3

G2 = clique_union(C3, D3)

@test ne(G2) == 21

G3 = clique_union(G2, C3)

@test ne(G3) == 21 + 6*3*2 + 3
@test nv(G3) == 9

G4 = disjoint_union(G3, clique_union(C3, C3))
G5 = clique_union(G4, Graph(2))

struct CoveredGraph
  G::Graph
  partition::Vector{Int}
end

CoveredGraph(G::Graph) = CoveredGraph(G, ones(nv(G)))
CoveredGraph(n::Int) = CoveredGraph(Graph(n), ones(n))

Graphs.nv(CG::CoveredGraph) = nv(CG.G)
Graphs.ne(CG::CoveredGraph) = ne(CG.G)

disjoint_union(CG::CoveredGraph, CH::CoveredGraph) = 
  CoveredGraph(disjoint_union(CG.G, CH.G),
    vcat(CG.partition, CH.partition .+ maximum(CG.partition)))

clique_union(CG::CoveredGraph, CH::CoveredGraph) = 
  CoveredGraph(clique_union(CG.G, CH.G),
    vcat(CG.partition, CH.partition .+ maximum(CG.partition)))


C3 = CoveredGraph(cycle_graph(Graph, 3))
D3 = CoveredGraph(Graph(3))

G1 = disjoint_union(C3, D3)

@test ne(G1) == 3

G2 = clique_union(C3, D3)

@test ne(G2) == 21

G3 = clique_union(G2, C3)

@test ne(G3) == 21 + 6*3*2 + 3
@test nv(G3) == 9

G4 = disjoint_union(G3, clique_union(C3, C3))
G5 = clique_union(G4, CoveredGraph(2))
