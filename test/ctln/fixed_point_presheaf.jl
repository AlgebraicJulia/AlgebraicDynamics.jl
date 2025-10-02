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
    FPg = FP(g)
    h = C(4)
    FPh = FP(h)
    FPgh = FP(disjoint_union(g, h))
    FPgh_dc = FPg + FPh # disjoint_union
    @test FPgh == FPgh_dc

    g = erdos_renyi(Graph, 7, 0.3)
    FPg = FP(g)
    h = C(4)
    FPh = FP(h)
    FPgh = FP(disjoint_union(g, h))
    FPgh_rand = FPg + FPh # disjoint_union
    @test sort(FPgh.data) == sort(FPgh_rand.data)

end

@testset "Multiple Disjoint Union" begin
    
    g1 = erdos_renyi(Graph, 7, 0.3)
    FPg1 = FP(g1) # FP(g1)
    g2 = C(4)
    FPg2 = FP(g2)
    FPg12 = FP(disjoint_union(g1, g2)) # no empty set
    FPg12′ = disjoint_union(FPg1, FPg2) # disjoint_union
    @test sort(FPg12.data) == sort(FPg12′.data)
    
    g3 = K(3)
    FPg3 = FP(g3)
    FPg123′ = foldl(disjoint_union, [FPg1, FPg2, FPg3])
    FPg123 = FP(disjoint_union(g1, g2, g3)) # reduces
    @test sort(FPg123.data) == sort(FPg123′.data)

end

@testset "Multiple Clique Union" begin
    
    g1 = erdos_renyi(Graph, 7, 0.3)
    FPg1 = FP(g1) # FP(g1)
    g2 = C(4)
    FPg2 = FP(g2)
    FPg12 = FP(clique_union(g1, g2))
    FPg12′ = clique_union(FPg1, FPg2)
    @test sort(FPg12.data) == sort(FPg12′.data)
    

    g3 = K(3)
    FPg3 = FP(g3)
    FPg123′ = foldl(clique_union, [FPg1, FPg2, FPg3])
    FPg123 = FP(clique_union(g1, g2, g3)) # reduces
    @test sort(FPg123.data) == sort(FPg123′.data)

end

@testset "Clique and Disjoint Union" begin
    
    g1 = erdos_renyi(Graph, 7, 0.3)
    FPg1 = FP(g1)
    g2 = C(4)
    FPg2 = FP(g2)
    g3 = K(3)
    FPg3 = FP(g3)
    FPg123′ = clique_union(disjoint_union(FPg1, FPg2), FPg3)
    FPg123 = FP(clique_union(disjoint_union(g1, g2), g3)) # reduces
    @test sort(FPg123.data) == sort(FPg123′.data)
    @test FPg123.base == FPg123′.base

end

# Now we need to test that we can construct an expression tree
@testset "Gluing Rule Expression Tree" begin
   
    # as usual, we compute the FP supports for two graphs, and show
    # we can compute the FP supports of their disjoint union.
    FPg1 = erdos_renyi(Graph, 7, 0.3) |> FP
    FPg2 = CycleGraph(100) |> FP
    FPg12 = disjoint_union(FPg1, FPg2)
    
    # Terminal lifts the FixedPointFunctors over g1 and g2 resp., into
    # terms in a gluing rule. This means we can now hold combinations of
    # terms in suspension until we are ready to compute them
    tFPg12 = FPg1 + FPg2
    FPg12′ = FP(tFPg12) # FixedPointSupports 

    # We show that the disjoint union we incrementally computed is equal
    # to the disjoint union expression we evaluated
    @test FPg12 == FPg12′

    # We attempt this for a more complicated expression
    FPg3 = CompleteGraph(3) |> FP
    FPg123 = clique_union(FPg3, disjoint_union(FPg1, FPg2))
   
    tFPg123 = FPg3 * (FPg1 + FPg2)
    FPg123′ = FP(tFPg123)

    @test FPg123 == FPg123′

    FPg12 = cyclic_union(FPg1, FPg2)
    FPg12′ = FP(FPg1 ↻ FPg2)

    @test FPg12 == FPg12′

end
