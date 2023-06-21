using Test

@testset "OrdinaryDiffEq Extension" begin
    include("OrdinaryDiffEq.jl")
end

@testset "DelayDiffEq Extension" begin
    include("DelayDiffEq.jl")
end