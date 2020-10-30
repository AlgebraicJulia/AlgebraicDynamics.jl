using AlgebraicDynamics
using AlgebraicDynamics.DiscDynam

using Catlab
using Catlab.WiringDiagrams
using Catlab.CategoricalAlgebra
using Catlab.CategoricalAlgebra.CSets
using Catlab.Programs.RelationalPrograms

import AlgebraicDynamics.DiscDynam.functor

using Plots

α = 1.2
β = 0.1
γ = 1.3
δ = 0.1

h = 0.1
gen = Dict(
    :birth     => u -> u .+ h .* [ α*u[1]],
    :death     => u -> u .+ h .* [-γ*u[1]],
    :predation => u -> u .+ h.* [-β*u[1]*u[2], δ*u[1]*u[2]],
)

d = @relation (x,y) where (x, y) begin
    birth(x)
    predation(x,y)
    death(y)
end

lv = functor( Dict(:birth => Dynam(gen[:birth], 1, [1], [0]),
                    :death => Dynam(gen[:death], 1, [1], [0]),
                    :predation => Dynam(gen[:predation], 2, [1,2], [0,0])))(d)

n = 100
u = zeros(Float64, 4, n)
u[:,1] = [17.0, 17.0, 11.0, 11.0]
for i in 1:n-1
    @views update!(u[:, i+1], lv, u[:, i])
    println(u[:, i+1])
end

p = plot(u')
savefig(p, "lvsoln.png")