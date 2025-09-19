using Catlab.Graphs
using Catlab.CategoricalAlgebra
using AlgebraicDynamics
using AlgebraicDynamics.ThresholdLinear
using Test

@test Support([1e-12, 1e-11]) == Support([2])
@test Support([1e-13]) == Support()

C3 = Term(C(3))
D3 = Term(D(3))

G1 = C3 * D3

@test ne(G1) == 3

G2 = C3 + D3

@test ne(G2) == 21

G3 = G2 + C3

@test ne(G3) == 21 + 6*3*2 + 3
@test nv(G3) == 9

G4 = G3 * (C3 * C3)
G5 = G4 + Term(Graph(2))

cC3 = CoveredGraph(C3)
cD3 = CoveredGraph(D3)

G1 = cC3 * cD3

@test ne(G1) == 3

G2 = C3 + cD3

@test ne(G2) == 21

G3 = G2 + cC3

@test ne(G3) == 21 + 6*3*2 + 3
@test nv(G3) == 9

G4 = G3 + (cC3 * cC3)
G5 = G4 + CoveredGraph(2)
