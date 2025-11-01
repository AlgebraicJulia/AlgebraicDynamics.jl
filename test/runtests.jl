using Test

@testset "UWDDynam" begin
  include("wd_dynam/uwd_dynam.jl")
end

@testset "DWDDynam" begin
  include("wd_dynam/dwd_dynam.jl")
end

@testset "CPGDynam" begin
  include("wd_dynam/cpg_dynam.jl")
end

@testset "ThresholdLinear" begin
  include("ctln/ThresholdLinear.jl")
end

@testset "Extensions" begin
  include("ext/extensions.jl")
end
