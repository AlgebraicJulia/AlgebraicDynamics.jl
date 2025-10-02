using AlgebraicDynamics
using AlgebraicDynamics.ThresholdLinear
using Catlab
using Catlab.Graphs
using Catlab.Graphics.Graphviz: view_graphviz, to_graphviz
using Test

@testset "Supports" begin
    @test Support([1,2,3.0]) == Support([1,2,3])
    @test Support([1,0,4.0]) == Support([1,3])
    @test Support([1,1e-15,1e-11]) == Support([1,3])
end

trig = @acset Graph begin
  V = 3
  E = 3
  src = [1,2,3]
  tgt = [2,3,1]
end

tri = TLNetwork(CTLNetwork(trig))
@test tri.W' == [ 0.0   -0.75  -1.5;
          -1.5    0.0   -0.75;
          -0.75  -1.5    0.0;
      ]

# Testing the enumeration of supports
G = Graph(4)
add_edge!(G, 1, 2)
add_edge!(G, 2, 3)
add_edge!(G, 3, 1)
add_edge!(G, 3, 4)
netG = TLNetwork(CTLNetwork(G))

H = Graph(5)
add_edges!(H, [1, 1, 2, 3, 3, 4, 5, 5], [2, 4, 5, 2, 4, 5, 1, 3])
netH= TLNetwork(CTLNetwork(H))

# fixed point supports
@test Supports(netG) == Support.([[4],
                                    [1, 2, 3],
                                    [1, 2, 3, 4]
                                   ])

@test Supports(netH) == Support.([[1, 2, 5],
                                    [1, 4, 5],
                                    [2, 3, 5],
                                    [3, 4, 5],
                                    [1, 2, 3, 5],
                                    [1, 2, 4, 5],
                                    [1, 3, 4, 5],
                                    [2, 3, 4, 5],
                                    [1, 2, 3, 4, 5]])

# TODO restriction_fixed_point


@test K(1) + K(1) == K(2)

C2 = C(2)
C3 = C(3)
D2 = D(2)
D3 = D(3)

supp_C3 = Local(C3)
@test supp_C3 == Support.([[1,2,3]])

supp_D2 = Local(D2)    
@test supp_D2 == Support.([[1], [2], [1,2]])

supp_D3 = Local(D3)
@test supp_D3 == Support.([[1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]])

d2d2 = CliqueUnion(D2, D2)
d2d2_collected = collect(d2d2)
@test d2d2_collected == @acset Graph begin
    V=4; E=8; src=[1,3,1,4,2,3,2,4]; tgt=[3,1,4,1,3,2,4,2]
end

c2c2 = CliqueUnion(C2, C2)
c2c2_collected = collect(c2c2)
@test c2c2_collected == @acset Graph begin
    V=4; E=12; src=[1,2,3,4,1,3,1,4,2,3,2,4]; tgt=[2,1,4,3,3,1,4,1,3,2,4,2]
end

G1 = DisjointUnion(C3, D3)
g1 = @acset Graph begin
    V=6
    E=3
    src=[1,2,3]
    tgt=[2,3,1]
end
g1_collected = collect(G1)
v1 = vertices(g1_collected) |> last
e1 = edges(g1_collected) |> last

@test g1 == g1_collected
@test e1 == 3 

G2_disjoint = DisjointUnion(G1, C3)
g2_disjoint_collected = collect(G2_disjoint)

G2 = CliqueUnion(G1, C3)
g2_collected = collect(G2)
v2 = vertices(g2_collected) |> last # 9
e2 = edges(g2_collected) |> last # 42

@test e2 == e1 + last(edges(C3)) + 2*v1*v2

G3 = CliqueUnion(G2, C3)
G4 = DisjointUnion(G3, CliqueUnion(C3, C3))

# example of construction of a graph using motives
# disjoint union of C3 and D3, only 3 edges from C3
G1 = disjoint_union(C3, D3)
# clique union of the previous graph G1 and another cycle C3, adding 3 edges and 6*2*3 edges between the two cliques
G2 = clique_union(G1, C3)
# clique union of the previous graph G2 and another cylcle C3, adding 3 edges and 9*3*2 edges between the two cliques
G3 = clique_union(G2, C3)
# add another two C3 motives (3+3 nodes)
G4 = disjoint_union(G3, clique_union(C3, C3))  
# clique union of the previous graph G4 and a discrete graph D2, adding 2 edges and 21*2*3 edges between the two cliques
G5 = clique_union(G4, D2)     



d2d2 = DisjointUnion(Term.([D2, D2]))
