using AlgebraicDynamics
using LinearAlgebra
using Polynomials
using OrdinaryDiffEq
using Plots
import Base: +

f(du, u, p, t) = begin
    du[1] = p[1]*u[1]
end

prob = ODEProblem(f, [1.0], (0,1.0), [0.5])
sol = solve(prob, alg=Tsit5())

plt = plot(sol)

fitsol(sol, deg) = begin
    map(1:length(sol.u[1])) do i
        uᵢ = [sol.u[t][i]-sol.u[1][i] for t in 1:length(sol.t)]
        pᵢ = fit(Polynomial, sol.t, uᵢ, deg)
        return pᵢ
    end
end

polysolve(prob::ODEProblem, deg=1) = begin
    sol = solve(prob, Tsit5(), saveat=0.01)
    poly = fitsol(sol, deg)
    return sol, poly
end

println("Plotting exponential growth solution and poly fit")
poly = fitsol(sol, 3)[1]
@show poly
plot!(plt, poly)

vitaldynamics(du, u, p, t) = begin
    du[1] = p[1]*u[1]
    du[2] = p[2]*u[2]
end


vdprob = ODEProblem(vitaldynamics, [1,1], (0.0,1.0), [0.5, -0.5])
sol, poly = polysolve(vdprob, 3)
println("Plotting vital dynamics solution and poly fit")
plt = plot(sol)
map(poly) do p
    plot!(plt, p, extrema(sol.t)..., linestyle=:dot)
end
plt


predation(du, u, p, t) = begin
    du[1] = -p[1]*u[1]*u[2]
    du[2] =  p[2]*u[1]*u[2]
end

lotkavolterra(du, u, p, t) = begin
    du[1] = p[1]*u[1] - p[3]*u[1]*u[2]
    du[2] = p[2]*u[2] + p[4]*u[1]*u[2]
end


u₀ = [1,1.0]
tspan = (0.0,1.0)
p₁ = [0.5, -0.5]
p₂ = [0.6,  0.5]
predprob = ODEProblem(predation, u₀, (0.0,1.0), p₂)

sol₁, poly₁ = polysolve(vdprob,   3)
sol₂, poly₂ = polysolve(predprob, 3)
@show poly₁[1]
@show poly₁[2]
@show poly₂[1]
@show poly₂[2]

function +(f::Function, q::Array{Polynomial{T}, 1}) where T
    function sumfield(du, u, p, t)
        f(du, u, p, t)
        for i in 1:length(u)
            du[i] += q[i](t)
        end
        return du
    end
    return sumfield
end

prob₁¹ = ODEProblem(vitaldynamics + poly₂, u₀, (0.0, 1.0), p₁)
prob₂¹ = ODEProblem(predation     + poly₁, u₀, (0.0, 1.0), p₂)

sol₁¹, poly₁¹ = polysolve(prob₁¹, 3)
sol₂¹, poly₂¹ = polysolve(prob₂¹, 3)

@show poly₁¹[1]
@show poly₁¹[2]
@show poly₂¹[1]
@show poly₂¹[2]

plot(sol₁¹)
plot(sol₂¹)

# Δapprox = poly₁[1] + poly₂[1] - (poly₁¹[1]+poly₂¹[1])

plot(integrate(poly₁¹[1] + poly₂¹[1]), tspan...)

lvprob = ODEProblem(lotkavolterra, u₀, (0.0,1.0), vcat(p₁, p₂))
sol_lv, poly_lv = polysolve(lvprob, 3)
plot(sol_lv)


comparesol(sol_true, sol_poly, measure=norm) = begin
    norm(sol_true(sol_true.t) - sol_poly(sol_true.t)) / length(sol_true.t)
end
comparesol(sol_true, pvec::Vector{Polynomial{T}}, measure=norm) where T = begin
    dims = 1:length(sol_true.u[1])
    pvals = map(sol_true.t) do t
        [pvec[i](t) for i in dims] .+ sol_true.u[1]
    end
    norm(sol_true(sol_true.t).u .- pvals) / length(sol_true.t)
end

@show comparesol(sol_lv, sol₁)
@show comparesol(sol_lv, sol₁¹)
@show comparesol(sol_lv, sol₂)
@show comparesol(sol_lv, sol₂¹)

# function (p::Vector{Polynomial{Float64}})(t)
#     map(p) do pᵢ
#         pᵢ(t)
#     end
# end

function refinepoly(f₁, f₂, f₃, u₀, tspan, p, steps, deg=1)
    p₁, p₂, p₃ = p
    prob₁ = ODEProblem(f₁, u₀, tspan, p₁)
    prob₂ = ODEProblem(f₂, u₀, tspan, p₂)
    prob₃ = ODEProblem(f₃, u₀, tspan, p₃)

    sol₁, poly₁ = polysolve(prob₁, deg)
    sol₂, poly₂ = polysolve(prob₂, deg)
    sol₃, poly₃ = polysolve(prob₃, deg)
    ϵ₁ = 0
    ϵ₂ = 0
    for i in 1:steps
        # update the problem assuming the environment is polynomial on the boundary
        prob₁¹ = ODEProblem(f₁ + poly₂, u₀, tspan, p₁)
        prob₂¹ = ODEProblem(f₂ + poly₁, u₀, tspan, p₂)

        #solve each subsystem for its effect on the boundary assuming env is poly
        sol₁¹, poly₁¹ = polysolve(prob₁¹, deg)
        sol₂¹, poly₂¹ = polysolve(prob₂¹, deg)

        # test for convergence
        # this method doesn't converge to the correct solution
        @show ϵ₁′ = comparesol(sol₃, sol₁¹)
        @show ϵ₂′ = comparesol(sol₃, sol₂¹)
        # @show ϵ₁′- ϵ₁
        # @show ϵ₂′- ϵ₂

        @show comparesol(sol₃, integrate.(poly₁¹.+poly₂¹))

        # but it does converge to a fixed point.
        @show norm(poly₁ - poly₁¹)
        @show norm(poly₂ - poly₂¹)

        #update state of iteration
        poly₁ = poly₁¹
        poly₂ = poly₂¹
        ϵ₁ = ϵ₁′
        ϵ₂ = ϵ₂′
    end
end

refinepoly(vitaldynamics, predation, lotkavolterra, [1.0,1.0], (0.0,1.0), (p₁, p₂, vcat(p₁, p₂)), 5, 5)
