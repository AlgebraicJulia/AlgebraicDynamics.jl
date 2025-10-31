using AlgebraicDynamics
using Catlab

using Test

c3 = CycleGraph(3)
@test nv(c3) == 3
@test ne(c3) == 3
@test Graph(c3) == cycle_graph(Graph, 3)
@test shift(c3, 4) == CycleGraph(3, 4)

k4 = CompleteGraph(4)
@test nv(k4) == 4
@test ne(k4) == 4^2
@test Graph(k4) == complete_graph(Graph, 4)
@test shift(k4, 3) == CompleteGraph(4, 3)

d5 = DiscreteGraph(5)
@test nv(d5) == 5
@test ne(d5) == 0
@test Graph(d5) == Graph(5)
@test shift(d5, 2) == DiscreteGraph(5, 2)


@testset "Connected Union" begin

    g = complete_graph(Graph, 3)
    h = complete_graph(Graph, 4)
    gh = connected_union(g, h)
    @test adjacency_matrix(gh) == adjacency_matrix(h)

    ghgh = connected_union(gh, gh)
    @test adjacency_matrix(ghgh) == adjacency_matrix(gh)

    g = Graph(5)
    h = Graph(4)
    gh = connected_union(g, h)
    @test adjacency_matrix(gh) == adjacency_matrix(g) 

    g = path_graph(Graph, 5)
    h = path_graph(Graph, 3)
    gh = connected_union(g, h)
    @test adjacency_matrix(gh) == adjacency_matrix(g)

end
