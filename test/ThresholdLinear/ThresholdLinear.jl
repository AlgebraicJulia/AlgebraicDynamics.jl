using AlgebraicDynamics
using AlgebraicDynamics.ThresholdLinear
using Catlab
using Catlab.Graphs
using Test




@test support([1,2,3.0]) == [1,2,3]
@test support([1,0,3.0]) == [1,3]
@test support([1,1e-15,1e-11]) == [1,3]


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

@test enumerate_supports_TLN(netG) == [[4],
                                    [1, 2, 3],
                                    [1, 2, 3, 4]
                                    ]

@test enumerate_supports_TLN(netH) == [[1, 2, 5],
                                    [1, 4, 5],
                                    [2, 3, 5],
                                    [3, 4, 5],
                                    [1, 2, 3, 5],
                                    [1, 2, 4, 5],
                                    [1, 3, 4, 5],
                                    [2, 3, 4, 5],
                                    [1, 2, 3, 4, 5]]
