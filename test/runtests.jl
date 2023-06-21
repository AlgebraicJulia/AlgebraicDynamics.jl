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

@testset "DiffEqs Extensions" begin
  include("ext/extensions.jl")
end
