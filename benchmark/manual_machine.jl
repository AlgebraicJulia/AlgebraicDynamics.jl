using BenchmarkTools
using AlgebraicDynamics
using AlgebraicDynamics.CPortGraphs
import AlgebraicDynamics.CPortGraphs: draw

using Catlab
using Catlab.WiringDiagrams
using Catlab.WiringDiagrams.CPortGraphs
using  Catlab.CategoricalAlgebra
using  Catlab.CategoricalAlgebra.FinSets
import Catlab.CategoricalAlgebra: coproduct
import Catlab.WiringDiagrams: oapply
using Catlab.Graphs
import Catlab.Graphs: Graph
import Catlab.CategoricalAlgebra.CSets: migrate!
using Catlab.Graphics
using Catlab.Graphics.Graphviz
using Base.Iterators
using Catlab.Theories

# TODO: Make this more elegantly in-place (currently a little hacky)
RDA(α₀, α₁, α₂) = begin
  diffop(du, u,p,t) = begin du .+= α₁ .* (sum(p) .- u .* length(p)) end
  advop(du, u,p,t)  = begin du .+= α₂ .* (p .- u) end
  update(du, u, p, t) = begin
    du .= α₀ .* u
    diffop(du, u, p, t)
    advop(du, u, p[2], t)
  end
  fc = VectorField{Float64}(update,
                            (du,u)->du[1:2] .= u[1],
                            2,
                            1)
  fw = VectorField{Float64}(update, (du,u)->(du[1:3] .= u[1]), 3, 1)
  fm = VectorField{Float64}(update, (du,u)->(du[1:4] .= u[1]), 4, 1)
  return fc, fw, fm
end

function generate_sys(size; parameters=[0,0.1,0.5], nsteps=10, stepsize=0.005)
  
  n, depth = size
  l = 2^(depth-1)
  fc,fw,fm = RDA(parameters...)
  G = grid(n, l)
  shape = vcat(           flatten(([fc], repeated(fw, n-2), [fc])) |> collect,
               take(cycle(flatten(([fw], repeated(fm, n-2), [fw]))), n*(l-2)) |> collect,
                          flatten(([fc], repeated(fw, n-2), [fc])) |> collect)
  return fc, fw, fm, G, shape
end

function combine_sys(G, shape)
  return oapply(G, shape)
end

function initialize(size)
  n, depth = size
  l = 2^(depth-1)
  inputs = ones(n)
  x_range = round(Int, n/4):round(Int, n*3/4)
  initial_state = zeros(l,n)
  initial_state[:, x_range] .= 1
  return initial_state
end

function simulate_bench(size, F, initial_state, nsteps=4000, stepsize=0.005)
  n, depth = size
  l = 2^(depth-1)

  traj = simulate(F, nsteps, stepsize, reshape(initial_state, n*l), vcat(zeros(n), zeros(n)))
end

function benchmark()
  res = Dict{String, Dict{String, Real}}()
  nesting = 2
  res["nest_pre_comp"] = Dict{String, Real}()
  
  gen_sys_val = @timed generate_sys((2^nesting,nesting+1), parameters=[0.01, 0.1, 0.5])
  res["nest_pre_comp"]["generators"] = gen_sys_val[2] 
  fc, fw, fm, G, shape = gen_sys_val[1]
  
  comb_sys_val = @timed combine_sys(G, shape)
  F = comb_sys_val[1]
  res["nest_pre_comp"]["construction"] = comb_sys_val[2]
  
  init_val = @timed initialize((2^nesting,nesting+1))
  
  res["nest_pre_comp"]["init_sys"] = init_val[2]
  initial_state = init_val[1]
  sim_val = @timed simulate_bench((2^nesting,nesting+1), F, initial_state)
  res["nest_pre_comp"]["solving"] = sim_val[2]
  
  println(res["nest_pre_comp"]["generators"])
  println(res["nest_pre_comp"]["construction"])
  println(res["nest_pre_comp"]["init_sys"])
  println(res["nest_pre_comp"]["solving"])
  println("Finished nesting of pre_comp")
  
  
  for nesting in 2:6
    
    res["nest_$nesting"] = Dict{String, Real}()
    gen_sys_val = @timed generate_sys((2^nesting,nesting+1), parameters=[0.01, 0.1, 0.5])
    res["nest_$nesting"]["generators"] = gen_sys_val[2] 
    fc, fw, fm, G, shape = gen_sys_val[1]
    
    comb_sys_val = @timed combine_sys(G, shape)
    F = comb_sys_val[1]
    res["nest_$nesting"]["construction"] = comb_sys_val[2]
    
    init_val = @timed initialize((2^nesting,nesting+1))
    
    res["nest_$nesting"]["init_sys"] = init_val[2]
    initial_state = init_val[1]
    sim_val = @timed simulate_bench((2^nesting,nesting+1), F, initial_state)
    res["nest_$nesting"]["solving"] = sim_val[2]
  
    println(res["nest_$nesting"]["generators"])
    println(res["nest_$nesting"]["construction"])
    println(res["nest_$nesting"]["init_sys"])
    println(res["nest_$nesting"]["solving"])
    println("Finished nesting of $nesting")
  end
end
benchmark()
