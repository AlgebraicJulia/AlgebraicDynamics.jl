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
