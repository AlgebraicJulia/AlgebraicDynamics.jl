using AlgebraicDynamics.CPortGraphDynam
using AlgebraicDynamics.DWDDynam
using AlgebraicDynamics.CPortGraphDynam: draw, barbell, gridpath, grid, meshpath
using Catlab

using ComponentArrays

using Test
using PrettyTables
using Base.Iterators: repeated

d = barbell(2)
xs = [
    ContinuousMachine{Float64}(2,2,(u, x, p, t)->[x[1]*u[1], x[2]*u[2]], (u,p,t)->u),
    ContinuousMachine{Float64}(2,2,(u, x, p, t)->[1/x[1]*u[1], -x[2]*u[2]], (u,p,t)->u)
]
h = 0.1
u₀ = ones(Float64, 4)
composite = oapply(d, xs)
fcomp(u,p,t) = eval_dynamics(composite, u, Float64[], p, t)
rcomp(u) = readout(composite,u)
@test rcomp(u₀) == []
u₁ = fcomp(u₀, Float64[], h)
@test u₁ == [1,1, 1, -1]
@test rcomp(u₁) == []
@test fcomp(u₁, [], h) == [1,-1, 1, 1]
# @show simulate(composite, 10, h, u₀)

d₀ = OpenCPortGraph()
add_parts!(d₀, :Box, 1)
d₁ = barbell(2)
F = ACSetTransformation((Box=[2],), d₀, d₁)
G = ACSetTransformation((Box=[1],), d₀, d₁)
# |1| <-> |3| <-> |5|
# |2| <-> |4| <-> |6|
d₂ = apex(pushout(F,G))
Catlab.Theories.id(OpenCPortGraph, n) = begin
    g = OpenCPortGraph()
    add_parts!(g, :Box, 1)
    add_parts!(g, :Port, n, box=1)
    add_parts!(g, :OuterPort, n, con=1:n)
    return g
end
lob(n) = let
    b = barbell(n)
    p = add_parts!(b, :Port, n, box=1)
    add_parts!(b, :OuterPort, n, con=p)
    b
end
d₂′ = ocompose(barbell(2), [id(OpenCPortGraph, 2), lob(2)])
# @show d₂′
β = 0.4
μ = 0.4
α₁ = 0.01
α₂ = 0.01

sirfuncb = (u,x,p,t)->[-β*u[1]*u[2] - α₁*(u[1]-x[1]), # Ṡ
                        β*u[1]*u[2] - μ*u[2] - α₂*(u[2]-x[2]), #İ
                        μ*u[2] # Ṙ
                        ]
sirfuncm = (u,x,p,t)->[-β*u[1]*u[2] - α₁*(u[1]-(x[1]+x[3])/2),
                        β*u[1]*u[2] - μ*u[2] - α₂*(u[2]-(x[2]+x[4])/2),
                        μ*u[2]
                        ]

boundary  = ContinuousMachine{Float64}(2,3,sirfuncb, (u,p,t)->u[1:2])
middle    = ContinuousMachine{Float64}(4,3, sirfuncm, (u,p,t)->u[[1,2,1,2]])
threecity = oapply(d₂, [boundary,middle,boundary])

# println("Simulating 3 city")
#traj = simulate(threecity, 100, 0.01, [100,1,0,100,0,0,100,0,0.0] )
# map(traj) do u
#     return (i1=u[2], i2=u[5], i3=u[8])
# end |> pretty_table

gl = @acset OpenCPortGraph begin
    Box = 3
    Port = 7
    Wire = 4
    OuterPort = 3
    box = [1,1,2,2,2,3,3]
    src = [2, 3, 5, 6]
    tgt = [3, 2, 6, 5]
    con = [1,4,7]
end

gm = @acset OpenCPortGraph begin
    Box = 3
    Port = 10
    Wire = 4
    OuterPort = 6
    box = [1,1,1,2,2,2,2,3,3,3]
    src = [2, 4, 6, 8]
    tgt = [4, 2, 8, 6]
    con = [3, 7, 10, 1, 5, 9]
end

gr = @acset OpenCPortGraph begin
    Box = 3
    Port = 7
    Wire = 4
    OuterPort = 3
    box = [1,1,2,2,2,3,3]
    src = [1, 3, 4, 6]
    tgt = [3, 1, 6, 4]
    con = [2,5,7]
end

symedges(g) = g.tables.W[g.tables.W.src .<= g.tables.W.tgt]
sympairs(z) = Iterators.filter(x->x[1] <= x[2], z)
pg2 = ocompose(barbell(3), [gl, gm])
g2 = migrate!(Catlab.Graphs.Graph(), pg2)
@test g2[ 9:11, :src] == [1,2,3]
@test g2[ 9:11, :tgt] == [4,5,6]
@test g2[12:14, :src] == [4,5,6]
@test g2[12:14, :tgt] == [1,2,3]

d3 = ocompose(barbell(3), [id(OpenCPortGraph, 3), lob(3)])
pg3 = ocompose(d3, [gl,gm,gr])
g3 = migrate!(Catlab.Graphs.Graph(), pg3)
@test g3[19:24, :src] == 1:6
@test g3[19:24, :tgt] == [4,5,6,1,2,3]
@test incident(pg3, 5, :box) == 11:14

α₁ = 1
fm = ContinuousMachine{Float64}(4, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 4)))
fl = ContinuousMachine{Float64}(3, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 3)))
fr = ContinuousMachine{Float64}(3, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 3)))
ft = ContinuousMachine{Float64}(3, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 3)))
fb = ContinuousMachine{Float64}(3, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 3)))
fc = ContinuousMachine{Float64}(2, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 2)))

@test eval_dynamics(ft, ones(1), ones(3), nothing, 0.0) == zeros(1)
f₁ = oapply(gl, [fc, fl, fc])
f₂ = oapply(gm, [ft, fm, fb])
f₃ = oapply(gr, [fc, fr, fc])

F = oapply(pg3, [fc, fl, fc, ft, fm, fb, fc, fr, fc])
@test eval_dynamics(F, ones(Float64, 9), [], nothing, 1.0) == zeros(Float64, 9)
u₀ = zeros(Float64, 9)
u₀[5] = 1.0
@test eval_dynamics(F, u₀, [], nothing, 1.0)[5] < 0
@test eval_dynamics(F, u₀, [], nothing, 1.0)[4] == α₁
@test eval_dynamics(F, u₀, [], nothing, 1.0)[2] == α₁
@test eval_dynamics(F, u₀, [], nothing, 1.0)[3] == 0

d4 = @acset OpenCPortGraph begin
    Box = 3
    Port = 18
    Wire = 12
    OuterPort = 6
    box = Iterators.flatten([repeated(i, 6) for i in 1:3])
    con = [1,2,3,16,17,18]
    src = [4,5,6,7,8,9,10,11,12,13,14,15]
    tgt = [7,8,9,4,5,6,13,14,15,10,11,12]
end
pg3 = ocompose(d4, [gm,gm,gm])
draw(pg3)
# pg3[:, :con]


@testset "Grids" begin
    @testset for (n, m) in Iterators.product(1:6, 2:4)
        g = grid(n, m)
        @test nparts(g, :Box) == n*m
        @test nparts(g, :Port) == 6n + 4n*(m-2)
        @test nparts(g, :Wire) == 2*((n-1)*m + (m-1)*n)
    end
end

