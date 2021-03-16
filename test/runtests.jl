using Test

@testset "UWDDynam" begin
  include("uwd_dynam.jl")
end

@testset "DWDDynam" begin
  include("dwd_dynam.jl")
end

@testset "CPGDynam" begin
  include("cpg_dynam.jl")
end

@testset "Trajectories and Solutions" begin
  include("trajectories.jl")
end

@testset "Tonti Diagrams" begin
  include("TontiDiagrams.jl")
end