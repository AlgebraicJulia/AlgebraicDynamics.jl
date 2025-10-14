using AlgebraicDynamics
using AlgebraicDynamics.ThresholdLinear
using Catlab
using Catlab.Graphs
using Catlab.Graphics.Graphviz: view_graphviz, to_graphviz
using Test

include("graph_utils.jl")
include("supports.jl")
include("fixed_point_presheaf.jl")

# trig = @acset Graph begin
#   V = 3
#   E = 3
#   src = [1,2,3]
#   tgt = [2,3,1]
# end

# tri = TLNetwork(CTLNetwork(trig))
# @test tri.W' == [ 0.0   -0.75  -1.5;
#           -1.5    0.0   -0.75;
#           -0.75  -1.5    0.0;
#       ]

# # Testing the enumeration of supports
# G = Graph(4)
# add_edge!(G, 1, 2)
# add_edge!(G, 2, 3)
# add_edge!(G, 3, 1)
# add_edge!(G, 3, 4)
# netG = TLNetwork(CTLNetwork(G))

# H = Graph(5)
# add_edges!(H, [1, 1, 2, 3, 3, 4, 5, 5], [2, 4, 5, 2, 4, 5, 1, 3])
# netH= TLNetwork(CTLNetwork(H))

# # fixed point supports
# @test Supports(netG) == Support.([[4],
#                                     [1, 2, 3],
#                                     [1, 2, 3, 4]
#                                    ])

# @test Supports(netH) == Support.([[1, 2, 5],
#                                     [1, 4, 5],
#                                     [2, 3, 5],
#                                     [3, 4, 5],
#                                     [1, 2, 3, 5],
#                                     [1, 2, 4, 5],
#                                     [1, 3, 4, 5],
#                                     [2, 3, 4, 5],
#                                     [1, 2, 3, 4, 5]])
