using Test

@testset "DWDDynam DiffEq Integration" begin
    include("DWDDynamDiffEqExt.jl")
end

@testset "UWDDynam DiffEq Integration" begin
    include("UWDDynamDiffEqExt.jl")
end

@testset "Trajectories DiffEq Integration" begin
    include("trajectories.jl")
end