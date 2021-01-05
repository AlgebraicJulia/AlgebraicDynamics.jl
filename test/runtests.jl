using Test

@testset "UWDDynam" begin
  include("uwd_dynam.jl")
end

@testset "DWDDynam" begin
  include("dwd_dynam.jl")
end

@testset "CPGDynam" begin
  include("cportgraphs.jl")
end