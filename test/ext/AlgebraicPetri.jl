using AlgebraicDynamics
using AlgebraicPetri
using Catlab
using Test

@testset "AlgebraicPetri" begin 
    birth_petri = Open(PetriNet(1, 1=>(1,1)));
    rs = ContinuousResourceSharer{Float64}(birth_petri)
    @test nstates(rs) == 1
    @test nports(rs) == 1
    @test portmap(rs) == [1]
    u = [1.0]; p = [2.0];
    @test eval_dynamics(rs, u, p, 0) == [2.0]
  
    Brusselator= LabelledPetriNet([:A, :B, :D, :E, :X, :Y], 
      :t1 => (:A => (:X, :A)),
      :t2 => ((:X, :X, :Y) => (:X, :X, :X)),
      :t3 => ((:B, :X) => (:Y, :D, :B)),
      :t4 => (:X => :E)
    )
  
    open_bruss = Open([:A, :D], Brusselator, [:A, :B])
    rs = ContinuousResourceSharer{Float64}(open_bruss)
    @test nstates(rs) == 6
    @test nports(rs) == 4
    @test portmap(rs) == [1,3, 1,2]
  end