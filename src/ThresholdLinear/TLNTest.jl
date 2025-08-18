using AlgebraicDynamics
using Catlab
using SciMLBase: NonlinearProblem, solve

g = erdos_renyi(Graph, 10, 0.3)

net = CTLNetwork(g)

fp_prob = NonlinearProblem(TLNetwork(net), ones(nv(g)) ./ nv(g))

restriction_fixed_point(g, collect(1:nv(g)))
