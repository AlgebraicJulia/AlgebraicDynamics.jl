using BenchmarkTools
include("../examples/diffusion/diffusion.jl")


function generate_sys()
  Δx = 0.1587 / 2
  h  = 0.005
  δ  = 0.01
  δ1  = 0.05
  v_x = 0.0
  γ = 0.0

  α=3/2*δ*h/(Δx^2)
  α1=3/2*δ1*h/(Δx^2)
  β=(1/4)*v_x*h/(Δx)
  κ=h*γ;

  gen = Dict(
      :diffuse_slw => u -> u+[(u[5]-u[1]*(1-β/α))*α,
                           (u[5]-u[2])*α,
                           (u[5]-u[3]*(1+β/α))*α,
                           (u[5]-u[4])*α,
                           (u[1]*(1-β/α) + u[2] + u[3]*(1+β/α) + u[4] - 4*(u[5]))*α + κ*u[5]],
      :diffuse_fst => u -> u+[(u[5]-u[1]*(1-β/α1))*α1,
                           (u[5]-u[2])*α1,
                           (u[5]-u[3]*(1+β/α1))*α1,
                           (u[5]-u[4])*α1,
                           (u[1]*(1-β/α1) + u[2] + u[3]*(1+β/α1) + u[4] - 4*u[5])*α1+ κ*u[5]]
  )

  # Turn generators into Open Dynam objects
  # These systems will look like the following
  #    n
  #    |
  # w- s -e
  #    |
  #    s

  diffuse_slw = Open(Dynam(gen[:diffuse_slw], 5, [1,2,3,4], [0,0,0,0,0]))
  diffuse_fst = Open(Dynam(gen[:diffuse_fst], 5, [1,2,3,4], [0,0,0,0,0]));
  return diffuse_slw, diffuse_fst
end

function combine_sys(diffuse_slw, diffuse_fst, nesting=6)
  #    n1   n2
  #    |    |
  # w1-d1-n-d2-e1
  #    |    |
  #    w    e
  #    |    |
  # w2-d3-s-d4-e2
  #    |    |
  #    s1   s2

  quad_hom = @relation (e1, e2, n1, n2, w1, w2, s1, s2) where (e1, e2, n1, n2, w1, w2, s1, s2, e, n, w, s) begin
      diffuse(n, n1, w1, w)
      diffuse(e1, n2, n, e)
      diffuse(s, w, w2, s1)
      diffuse(e2, e, s, s2)
  end

  #    n1   n2
  #    |    |
  # w1-m2-n-m1-e1
  #    |    |
  #    w    e
  #    |    |
  # w2-m1-s-m1-e2
  #    |    |
  #    s1   s2

  quad_qtr = @relation (e1, e2, n1, n2, w1, w2, s1, s2) where (e1, e2, n1, n2, w1, w2, s1, s2, e, n, w, s) begin
      diffuse_m2(n, n1, w1, w)
      diffuse_m1(e1, n2, n, e)
      diffuse_m1(s, w, w2, s1)
      diffuse_m1(e2, e, s, s2)
  end;

  #    n1   n2
  #    |    |
  # w1-m1-n-m2-e1
  #    |    |
  #    w    e
  #    |    |
  # w2-m2-s-m1-e2
  #    |    |
  #    s1   s2

  quad_hlf = @relation (e1, e2, n1, n2, w1, w2, s1, s2) where (e1, e2, n1, n2, w1, w2, s1, s2, e, n, w, s) begin
      diffuse_m1(n, n1, w1, w)
      diffuse_m2(e1, n2, n, e)
      diffuse_m2(s, w, w2, s1)
      diffuse_m1(e2, e, s, s2)
  end;

  #    n1   n2
  #    |    |
  # w1-m1-n-m2-e1
  #    |    |
  #    w    e
  #    |    |
  # w2-m3-s-m4-e2
  #    |    |
  #    s1   s2

  quad_all = @relation (e1, e2, n1, n2, w1, w2, s1, s2) where (e1, e2, n1, n2, w1, w2, s1, s2, e, n, w, s) begin
      diffuse_m1(n, n1, w1, w)
      diffuse_m2(e1, n2, n, e)
      diffuse_m3(s, w, w2, s1)
      diffuse_m4(e2, e, s, s2)
  end;

  periodic_bcs = @relation (n, s) where (n, s, ew) begin
      diffuse(ew, n, ew, s)
  end;

  # Homogenous 2^n x 2^n system of metamaterial 1
  diff_mm1 = functor(quad_qtr, Dict(:diffuse_m1=>diffuse_fst, :diffuse_m2=>diffuse_slw), bundling=[[1,2],[3,4],[5,6],[7,8]]);
  for i in 1:(nesting - 2)
    diff_mm1 = functor(quad_hom, Dict(:diffuse=>diff_mm1), bundling=[[1,2],[3,4],[5,6],[7,8]]);
  end

  # Homogenous 2^n x 2^n system of metamaterial 2
  diff_mm2 = functor(quad_qtr, Dict(:diffuse_m1=>diffuse_slw, :diffuse_m2=>diffuse_fst), bundling=[[1,2],[3,4],[5,6],[7,8]]);
  for i in 1:(nesting - 2)
    diff_mm2 = functor(quad_hom, Dict(:diffuse=>diff_mm2), bundling=[[1,2],[3,4],[5,6],[7,8]]);
  end

  # Heterogenous 32x32 system of material 1 and material 2 in checkerboard pattern
  diff = functor(quad_hlf, Dict(:diffuse_m1=>diff_mm2, :diffuse_m2=>diff_mm1), bundling=[[1,2],[3,4],[5,6],[7,8]]);
  #diff_64x64 = functor(quad_hom, Dict(:diffuse=>diff_32x32_m1), bundling=[[1,2],[3,4],[5,6],[7,8]]);
  diff_periodic = functor(periodic_bcs, Dict(:diffuse=>diff));
  return diff_periodic
end

function initialize(nests=6)
  c_size = 2^nests
  coords = get_coords(nests)*5
  state = zeros(length(coords) * 5)
  x_range = round(Int, c_size/4):round(Int, c_size*3/4)
  state[coords[x_range, :]] .= ones(length(x_range), size(coords,1))*5
  state = interp_vals(state[coords], coords)
  return state, coords
end

function simulate(diff_64x64_periodic, state, coords)
  res = zeros(length(state))
  #state_array = [state]
  for i ∈ 1:4000
      view(state, coords[:,end] .- 1) .= view(state, coords[:,end] .- 3)
      view(state, coords[:,1] .- 3) .= view(state, coords[:,1] .- 1)
      update!(res, diff_64x64_periodic.cospan.apex, state);
      state .= res;
      #push!(state_array, state)
  end;
end

res = Dict{String, Dict{String, Real}}()
#precomp run
nesting = 2
res["nest_pre_comp"] = Dict{String, Real}()

gen_sys_val = @timed generate_sys()
res["nest_pre_comp"]["generators"] = gen_sys_val[2] 
diff_slw, diff_fst = gen_sys_val[1]

comb_sys_val = @timed combine_sys(diff_slw, diff_fst, nesting)
system = comb_sys_val[1]
res["nest_pre_comp"]["construction"] = comb_sys_val[2]

init_val = @timed initialize(nesting) 

res["nest_pre_comp"]["init_sys"] = init_val[2]
state, coords = init_val[1]
sim_val = @timed simulate(system, state, coords)
res["nest_pre_comp"]["solving"] = sim_val[2]

println(res["nest_pre_comp"]["generators"])
println(res["nest_pre_comp"]["construction"])
println(res["nest_pre_comp"]["init_sys"])
println(res["nest_pre_comp"]["solving"])
println("Finished nesting of pre_comp")


for nesting in 2:6
  res["nest_$nesting"] = Dict{String, Real}()

  gen_sys_val = @timed generate_sys()
  res["nest_$nesting"]["generators"] = gen_sys_val[2] 
  diff_slw, diff_fst = gen_sys_val[1]

  comb_sys_val = @timed combine_sys(diff_slw, diff_fst, nesting)
  system = comb_sys_val[1]
  res["nest_$nesting"]["construction"] = comb_sys_val[2]

  init_val = @timed initialize(nesting) 

  res["nest_$nesting"]["init_sys"] = init_val[2]
  state, coords = init_val[1]
  sim_val = @timed simulate(system, state, coords)
  res["nest_$nesting"]["solving"] = sim_val[2]

  println(res["nest_$nesting"]["generators"])
  println(res["nest_$nesting"]["construction"])
  println(res["nest_$nesting"]["init_sys"])
  println(res["nest_$nesting"]["solving"])
  println("Finished nesting of $nesting")
end
