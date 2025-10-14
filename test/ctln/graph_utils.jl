using AlgebraicDynamics

using Test

c3 = CycleGraph(3)
@test nv(c3) == 3
@test ne(c3) == 3
@test Graph(c3) == C(3)
@test shift(c3, 4) == CycleGraph(3, 4)

k4 = CompleteGraph(4)
@test nv(k4) == 4
@test ne(k4) == 4^2
@test Graph(k4) == K(4)
@test shift(k4, 3) == CompleteGraph(4, 3)

d5 = DiscreteGraph(5)
@test nv(d5) == 5
@test ne(d5) == 0
@test Graph(d5) == D(5)
@test shift(d5, 2) == DiscreteGraph(5, 2)
