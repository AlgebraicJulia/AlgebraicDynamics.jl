using Test
using Combinatorics: powerset
using Catlab: nv, Graph, erdos_renyi
using AlgebraicDynamics
using AlgebraicDynamics.ThresholdLinear: disjoint_union

@testset "Disjoint Union" begin
    
    g = Graph(3)
    FPg = FP(g)
    h = cycle_graph(Graph, 4)
    FPh = FP(h)
    FPgh = FP(disjoint_union(g, h))
    FPgh_dc = FP(FPg + FPh) # disjoint_union
    @test FPgh == FPgh_dc

    g = erdos_renyi(Graph, 7, 0.3)
    FPg = FP(g)
    h = cycle_graph(Graph, 4)
    FPh = FP(h)
    FPgh = FP(disjoint_union(g, h))
    FPgh_rand = FP(FPg + FPh) # disjoint_union
    @test sort(FPgh.data) == sort(FPgh_rand.data)

end

@testset "Connected Union" begin

    # idempotency
    g = complete_graph(Graph, 4)
    FPg = FP(g)
    FPgh = FP(connected_union(g, g))
    FPgh_dc = connected_union(FPg, FPg)
    @test FPgh == FPgh_dc

    # fixed-point functor preserves connected union of complete graphs
    g = complete_graph(Graph, 3)
    FPg = FP(g)
    h = complete_graph(Graph, 4)
    FPh = FP(h)
    FPgh = FP(connected_union(g, h))
    FPgh_dc = connected_union(FPg, FPh)
    @test FPgh == FPgh_dc

    # fixed point functor preserves connected union of path graphs
    g = path_graph(Graph, 2)
    FPg = FP(g)
    h = path_graph(Graph, 3)
    FPh = FP(h)
    FPgh = FP(connected_union(g, h))
    FPgh_dc = connected_union(FPg, FPh)
    @test FPgh == FPgh_dc

    # fixed point functor preserves connected union of discrete and path graphs
    i = Graph(1)
    FPi = FP(i)
    h = path_graph(Graph, 3)
    FPh = FP(h)
    FPgh = FP(connected_union(i, h))
    FPgh_dc = connected_union(FPi, FPh)
    @test FPgh == FPgh_dc

    @test ∪(FPi, FPg, FPh) == connected_union(connected_union(FPi, FPg), FPh)
    @test FP(↻(FPi, FPg, FPh)) == cyclic_union(cyclic_union(FPi, FPg), FPh)

    # This should fail because a1 and a2 is not a simply embedded cover of `g` 
    g = path_graph(Graph, 16)
    a1 = Subobject(g, V=1:10, E=1:9)
    a2 = Subobject(g, V=9:16, E=9:15)
    @test FP(g) != connected_union(FP(a1), FP(a2))

    # This should pass because a1 and a2 are a simply embedded cover of `g` 
    g = @acset Graph begin
        V=3
        E=2
        src=[1,2]
        tgt=[3,3]
    end
    a1 = Subobject(g, V=[1,3], E=[1])
    a2 = Subobject(g, V=[2,3], E=[2])
    @test FP(g) == connected_union(FP(a1), FP(a2))

end

@testset "Multiple Disjoint Union" begin
    
    g1 = erdos_renyi(Graph, 7, 0.3)
    FPg1 = FP(g1) # FP(g1)
    g2 = cycle_graph(Graph, 4)
    FPg2 = FP(g2)
    FPg12 = FP(disjoint_union(g1, g2)) # no empty set
    FPg12′ = FP(FPg1 + FPg2) # disjoint_union
    @test sort(FPg12.data) == sort(FPg12′.data)
    
    g3 = Graph(3)
    FPg3 = FP(g3)
    FPg123′ = foldl(disjoint_union, [FPg1, FPg2, FPg3])
    FPg123 = FP(disjoint_union(g1, g2, g3)) # reduces
    @test sort(FPg123.data) == sort(FPg123′.data)

    @test FP(+(FPg1, FPg2, FPg3)) == disjoint_union(disjoint_union(FPg1, FPg2), FPg3)

end

@testset "Multiple Clique Union" begin
    
    g1 = erdos_renyi(Graph, 7, 0.3)
    FPg1 = FP(g1) # FP(g1)
    g2 = cycle_graph(Graph, 4)
    FPg2 = FP(g2)
    FPg12 = FP(clique_union(g1, g2))
    FPg12′ = clique_union(FPg1, FPg2)
    @test sort(FPg12.data) == sort(FPg12′.data)
    

    g3 = Graph(3)
    FPg3 = FP(g3)
    FPg123′ = foldl(clique_union, [FPg1, FPg2, FPg3])
    FPg123 = FP(clique_union(g1, g2, g3)) # reduces
    @test sort(FPg123.data) == sort(FPg123′.data)

    @test FP(*(FPg1, FPg2, FPg3)) == clique_union(clique_union(FPg1, FPg2), FPg3)

end

@testset "Clique and Disjoint Union" begin
    
    g1 = erdos_renyi(Graph, 7, 0.3)
    FPg1 = FP(g1)
    g2 = cycle_graph(Graph, 4)
    FPg2 = FP(g2)
    g3 = Graph(3)
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
    FPg12′ = FP(FPg1 + FPg2) # FixedPointSupports 

    # We show that the disjoint union we incrementally computed is equal
    # to the disjoint union expression we evaluated
    @test FPg12 == FPg12′

    # We attempt this for a more complicated expression
    FPg3 = CompleteGraph(3) |> FP
    FPg123 = clique_union(FPg3, disjoint_union(FPg1, FPg2))
    FPg123′ = FP(FPg3 * (FPg1 + FPg2)) 

    @test FPg123 == FPg123′

    FPg12 = cyclic_union(FPg1, FPg2)
    FPg12′ = FP(FPg1 ↻ FPg2)

    @test FPg12 == FPg12′

    FPg3c1p12 = cyclic_union(FPg3, disjoint_union(FPg1, clique_union(FPg2, FPg1)))
    FPg3c1p12′ = FP(FPg3 ↻ (FPg1 + (FPg2 * FPg1)))
    @test FPg3c1p12 == FPg3c1p12′

    @test FP(↻(↻(FPg1, FPg1), FPg1)) == FP(↻(FPg1, FPg1, FPg1))

end
