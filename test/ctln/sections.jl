using Test
using Combinatorics: powerset
using Catlab: nv, Graph, erdos_renyi
using AlgebraicDynamics


# @testset "Sections" begin
#     K3 = CompleteGraph(3)
#     K2 = CompleteGraph(2)
#     gh = K3 + K2
#     #
#     l1 = FPSections(K3) # computes supports for G|τᵢ
#     l2 = FPSections(K2)
#     @test l1 == FPSections([Support(1:3)])
#     @test l2 == FPSections([Support(1:2)])
#     #
#     l12 = l1 ∨ l2
#     lgh = FPSections(gh)
#     @test sort(l12) == sort(lgh)
# end

# @testset "Sections" begin
#     K8 = CompleteGraph(8)
#     D4 = DiscreteGraph(4)
#     gh = K8 + D4
#     #
#     l1 = FPSections(K8)
#     l2 = FPSections(collect(D4)) # TODO no collect
#     l1′ = FPSections([Support(1:8)])
#     l2′ = FPSections([Support(k) for k in powerset(1:nv(D4)) if !isempty(k)])
#     @test l1 == l1′ 
#     @test l2 == l2′ 
#     #
#     l12 = l1 ∨ l2
#     lgh = FPSections(gh)
#     @test sort(l12) == sort(lgh)
# end

# @testset "Random Graphs" begin

#     # these need to be promoted to Gluing Terms
#     g = erdos_renyi(Graph, 10, 0.3)
#     D4 = DiscreteGraph(4)
#     gh = DisjointUnion(g, D4)

# end

# g = CompleteGraph(3)
# shift!(g, 2)
# @test g.offset == 2

@testset "Disjoint Union" begin
    
    g = D(3)
    FPg = FPFunctor(g)
    h = C(4)
    FPh = FPFunctor(h)
    FPgh = FPFunctor(disjoint_union(g, h))
    FPgh_dc = FPg + FPh # disjoint_union
    @test FPgh == FPgh_dc

    g = erdos_renyi(Graph, 7, 0.3)
    FPg = FPFunctor(g)
    h = C(4)
    FPh = FPFunctor(h)
    FPgh = FPFunctor(disjoint_union(g, h))
    FPgh_rand = FPg + FPh # disjoint_union
    @test sort(FPgh.data) == sort(FPgh_rand.data)

end

@testset "Multiple Disjoint Union" begin
    
    g1 = erdos_renyi(Graph, 7, 0.3)
    FPg1 = FPFunctor(g1) # FPFunctor(g1)
    g2 = C(4)
    FPg2 = FPFunctor(g2)
    FPg12 = FPFunctor(disjoint_union(g1, g2)) # no empty set
    FPg12′ = FPg1 + FPg2 # disjoint_union
    @test sort(FPg12.data) == sort(FPg12′.data)
    
    g3 = K(3)
    FPg3 = FPFunctor(g3)
    FPg123′ = FPg1 + FPg2 + FPg3
    FPg123 = FPFunctor(disjoint_union(g1, g2, g3)) # reduces
    @test sort(FPg123.data) == sort(FPg123′.data)

end

@testset "Multiple Clique Union" begin
    
    g1 = erdos_renyi(Graph, 7, 0.3)
    FPg1 = FPFunctor(g1) # FPFunctor(g1)
    g2 = C(4)
    FPg2 = FPFunctor(g2)
    FPg12 = FPFunctor(clique_union(g1, g2))
    FPg12′ = FPg1 * FPg2
    @test sort(FPg12.data) == sort(FPg12′.data)
    

    g3 = K(3)
    FPg3 = FPFunctor(g3)
    FPg123′ = FPg1 * FPg2 * FPg3
    FPg123 = FPFunctor(clique_union(g1, g2, g3)) # reduces
    @test sort(FPg123.data) == sort(FPg123′.data)

end

@testset "Clique and Disjoint Union" begin
    
    g1 = erdos_renyi(Graph, 7, 0.3)
    FPg1 = FPFunctor(g1)
    g2 = C(4)
    FPg2 = FPFunctor(g2)
    g3 = K(3)
    FPg3 = FPFunctor(g3)
    FPg123′ = (FPg1 + FPg2) * FPg3
    FPg123 = FPFunctor(clique_union(disjoint_union(g1, g2), g3)) # reduces
    @test sort(FPg123.data) == sort(FPg123′.data)
    @test FPg123.base == FPg123′.base

end

# Now we need to test that we can construct an expression tree
@testset "Gluing Rule Expression Tree" begin
   
    # as usual, we compute the FP supports for two graphs, and show
    # we can compute the FP supports of their disjoint union.
    FPg1 = erdos_renyi(Graph, 7, 0.3) |> FPFunctor
    FPg2 = CycleGraph(100) |> FPFunctor
    FPg12 = FPg1 + FPg2
    
    # Terminal lifts the FixedPointFunctors over g1 and g2 resp., into
    # terms in a gluing rule. This means we can now hold combinations of
    # terms in suspension until we are ready to compute them
    tFPg1, tFPg2 = Terminal(FPg1), Terminal(FPg2)
    tFPg12 = tFPg1 + tFPg2
    FPg12′ = FPFunctor(tFPg12) # FixedPointSupports 
    # TODO this should be called FPFunctor rather than Graph

    # We show that the disjoint union we incrementally computed is equal
    # to the disjoint union expression we evaluated
    @test FPg12 == FPg12′

    # We attempt this for a more complicated expression
    FPg3 = CompleteGraph(3) |> FPFunctor
    FPg123 = FPg3 * (FPg1 + FPg2)
   
    tFPg3 = FPg3 |> Terminal
    tFPg123 = tFPg3 * (tFPg1 + tFPg2)
    FPg123′ = FPFunctor(tFPg123)

    @test FPg123 == FPg123′ 

end

# TODO implement FP which consumes these expressions and turns them into a new FP
CliqueUnion(FPgh_dc, FPgh_rand)

@testset "Disjoint Union" begin
    g = D(3)
    FPg = FPFunctor(g)
    h = C(4)
    FPh = FPFunctor(h)
    FPgh = FPFunctor(disjoint_union(g, h))
    FPgh′ = FPg + FPh
    @test FPgh == FPgh′
end

# Clique Union
g = D(1)
FPg = FPFunctor(g)
h = D(1)
gh = clique_union(g, h)
FPh = FPFunctor(h)
FPgh = FPFunctor(gh)
FPgh′ = FPg * FPh

g1 = erdos_renyi(Graph, 10, 0.4)
FPg1 = FPFunctor(g1)
g2 = erdos_renyi(Graph, 10, 0.4)
FPg2 = FPFunctor(g2)
g3 = erdos_renyi(Graph, 10, 0.4)
FPg3 = FPFunctor(g3)
FPg = FPg1 + (FPg2 * FPg3)

# ...
g = FPFunctor(CompleteGraph(3))
h = CompleteGraph(2)
gh = g * h
# supports are [1,2], [3,4], [1,2,3,4]

g
FPFunctor(h)
FPFunctor(gh)

i = FPFunctor(CompleteGraph(5))
ghi = gh * i


# ...
g = FPFunctor(CompleteGraph(3))
h = CycleGraph(5)
gh = g * h
# supports are [1,2], [3,4], [1,2,3,4]

g.data
FPFunctor(h).data
FPFunctor(gh).data


g = FPFunctor(collect(DiscreteGraph(2)))
