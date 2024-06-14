using Catlab
using Catlab.CategoricalAlgebra
using AlgebraicDynamics

import AlgebraicDynamics.UWDDynam: euler_approx

using AlgebraicDynamics
using Catlab.WiringDiagrams, Catlab.Programs

using LabelledArrays
using OrdinaryDiffEq, Plots, Plots.PlotMeasures

const UWD = UndirectedWiringDiagram

## Define the primitive systems
dotr(u,p,t) = p.α*u
dotrf(u,p,t) = [-p.β*u[1]*u[2], p.γ*u[1]*u[2]]
dotf(u,p,t) = -p.δ*u

rabbit_growth = ContinuousResourceSharer{Float64}(1, dotr)
rabbitfox_predation = ContinuousResourceSharer{Float64}(2, dotrf)
fox_decline = ContinuousResourceSharer{Float64}(1, dotf)

## Define the composition pattern
rf = @relation (rabbits,foxes) begin 
    growth(rabbits)
    predation(rabbits,foxes)
    decline(foxes)
end

## Compose
rabbitfox_system = oapply(rf, [rabbit_growth, rabbitfox_predation, fox_decline])

## Solve and plot
u0 = [10.0, 100.0]                              
params = LVector(α=.3, β=0.015, γ=0.015, δ=0.7)
tspan = (0.0, 100.0)    

prob = ODEProblem(rabbitfox_system, u0, tspan, params)
sol = solve(prob, Tsit5())

plot(sol, rabbitfox_system,
    lw=2, title = "Lotka-Volterra Predator-Prey Model",
    xlabel = "time", ylabel = "population size"
)


dt = 1.0
dsol₁  = solve(DiscreteProblem(euler_approx(rabbitfox_system, dt, 10), u0, tspan ./ dt , params), FunctionMap())
dsol₁₀ = solve(DiscreteProblem(euler_approx(rabbitfox_system, dt, 100), u0, tspan ./ dt, params), FunctionMap())
dsolfine = solve(DiscreteProblem(euler_approx(rabbitfox_system, dt, 1000), u0, tspan ./ dt, params), FunctionMap())
dsolextrafine = solve(DiscreteProblem(euler_approx(rabbitfox_system, dt, 10000), u0, tspan ./ dt, params), FunctionMap())

  p = plot(dsol₁, idxs=[1])
  plot!(p, dsol₁₀, idxs=[1])
  plot!(p, dsolfine, idxs=[1])
  plot!(p, dsolextrafine, idxs=[1])
  plot!(p, sol, idxs=[1])

begin
  dt = 0.05
  tspan′ = tspan ./ dt
  disc_rabbitfox_system_coarse = oapply(rf, euler_approx([rabbit_growth, rabbitfox_predation, fox_decline], dt, 1))
  disc_sol_coarse = solve(DiscreteProblem(disc_rabbitfox_system_coarse, u0, tspan′, params), FunctionMap())
  disc_rabbitfox_system = oapply(rf, euler_approx([rabbit_growth, rabbitfox_predation, fox_decline], dt, 10))
  disc_sol = solve(DiscreteProblem(disc_rabbitfox_system, u0, tspan′, params), FunctionMap())
  disc_rabbitfox_system_fine = oapply(rf, euler_approx([rabbit_growth, rabbitfox_predation, fox_decline], dt, 100))
  disc_sol_fine = solve(DiscreteProblem(disc_rabbitfox_system_fine, u0, tspan′, params), FunctionMap())
  disc_rabbitfox_system_extra_fine = oapply(rf, euler_approx([rabbit_growth, rabbitfox_predation, fox_decline], dt, 1000))
  disc_sol_extra_fine = solve(DiscreteProblem(disc_rabbitfox_system_extra_fine, u0, tspan′, params), FunctionMap())
  p = plot(disc_sol_extra_fine, idxs=[1], label="extrafine")
  plot!(p, disc_sol_fine, idxs=[1], label="fine")
  plot!(p, disc_sol, idxs=[1], label="solution")
  plot!(p, disc_sol_coarse, idxs=[1], label="coarse")
end

# p = plot(sol, idxs=[1])

function euler!(output, f, u0, tspan, params)
  ui = copy(u0)
  for t in tspan
    ui .= eval_dynamics(f, ui, params, t)
    push!(output, copy(ui))
  end
  return hcat(output...)'
end

sol₁  = euler!(Vector{Float64}[], euler_approx(rabbitfox_system, 0.01), u0, 1:3200, params)
sol₁₀ = euler!(Vector{Float64}[], euler_approx(rabbitfox_system, 0.01, 10), u0, 1:3200, params)

plot(sol₁)

plot(abs.(sol₁ .- sol₁₀) ./ sol₁₀)