module TontiTest
  using AlgebraicDynamics.Tonti
  using CombinatorialSpaces
  using GeometryBasics
  using Random
  using Test
  td = TontiDiagram(2, [:C =>:IP,  :dC =>:TP,  :ΔC=>:IL,
                         :∑ϕ=>:TS2, :ϕ=>:TL2]);

  s = EmbeddedDeltaSet2D{Bool, Point{3,Float64}}()
  points = [(0,0,0),(0,0,1),(0,1,0),(0,1,1)]

  add_vertices!(s, 4, point=points)
  glue_sorted_triangle!(s, 1,2,3)
  glue_sorted_triangle!(s, 2,3,4)

  sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point{3,Float64}}(s)
  subdivide_duals!(sd, Barycenter())
  star0 = ⋆(0,sd)

  @test gen_form(s, x->1) == [1,1,1,1]

  addSpace!(td, s)
  addTime!(td)
  const k = 0.003

  diffusion(diff) = ΔC -> diff*ΔC
  mass_conv() = ∑ϕ->∑ϕ

  addTransform!(td, s, [:ΔC], diffusion(k), [:ϕ])
  addTransform!(td, s, [:∑ϕ], mass_conv(), [:dC]);

  sim, data_syms = vectorfield(td, s)

  @test data_syms[1][1] == :C
  @test data_syms[1][2] == 4

  Random.seed!(42)
  u  = gen_form(s,x->rand())
  du = zeros(Float64,nv(s))

  tempu = copy(u)
  dt = 0.001
  for i in 10000
    sim(du, tempu, [],0)
    tempu .+= du
    # Check that mass is conserved
  end
  @test all(tempu .!= u)
  @test all(sum(star0*tempu)-sum(star0*u) < 1e-6)
end
