using DifferentialEquations
using DiffEqOperators
using Distributions

function generate_sys(sys_size=64)
  # Physical Constants
  α = 0.05 # Diffusion Coefficient
  v_u = 0 # Advection term for x-velocity
  v_v = 0 # Advection term for x-velocity
  γ = 0 # Source term
  
  # Discretization Parameters
  deriv_l = 2
  order_l = 2
  deriv_d = 1
  order_d = 2
  s = x, y = (range(0, stop=10, length=sys_size), range(0, stop=10, length=sys_size))
  dx = dy = x[2] - x[1]
  t0 = 0.0
  t1 = 20.0;
  
  # Initial Distribution
  u_low_x = round(Int64, 4 / 10 * length(x))
  u_high_x = round(Int64, 6 / 10 * length(x))
  u_low_y = round(Int64, 0.1 / 10 * length(x))
  u_high_y = round(Int64, 10 / 10 * length(x))
  
  u0 = fill(0.0, (length(x), length(y)));
  #u0[u_low_x:u_high_x,u_low_y:u_high_y] .= 5.0;   # a small square in the middle of the domain
  x_range = round(Int, sys_size/4):round(Int, sys_size*3/4)
  u0[x_range, :] .= 5.0
  
  # Coefficient functions
  # Diffusion Coefficient Matrix
  α_mid_x = round(Int64, 5 / 10 * length(x))
  α_mid_y = round(Int64, 5 / 10 * length(x))
  
  # Checkerboarded Diffusion Coefficient
  a = 0.05
  b = 0.01
  α_1 = fill(b,α_mid_x,α_mid_y)
  α_1[1:2:α_mid_x, 1:2:α_mid_y] .= a
  α_2 = fill(a,α_mid_x,α_mid_y)
  α_2[1:2:α_mid_x, 1:2:α_mid_y] .= b
  #α = hcat(vcat(α_2, α_2), vcat(α_2, α_2));
  α = hcat(vcat(α_1, α_2), vcat(α_2, α_1));

	return dx, dy, x, y, α, t0, t1, u0
end

function combine_sys(dx, dy, x, y)
  # Generate Coefficient Matrices
  laplacianx = CenteredDifference{1}(2, 2, dx, length(x))
  laplaciany = CenteredDifference{2}(2, 2, dy, length(y))
  Δ = laplacianx + laplaciany;
  
  divx = CenteredDifference{1}(1, 2, dx, length(x))
  divy = CenteredDifference{2}(1, 2, dy, length(y))
  ∇ = divx + divy
  
  # Generate Boundary Conditions
  Qx, Qy = Neumann0BC(Float64, (dx, dy), 2, (length(x), length(y)))
  
  ### Testing Arbitrary Boundary Conditions
  # Create atomic BC
  q1 = PeriodicBC(Float64)
  q2 = PeriodicBC(Float64)
  
  BCx = vcat(fill(q1, div(length(y), 2)), fill(q2, length(y) - div(length(y), 2)))  # The size of BCx has to be all size components *except* for x
  
  Qx = MultiDimBC{1}(BCx)
  
  Q = compose(Qx, Qy);
  return Δ, Q
end

function initialize(Δ, Q, α, t0, t1, u0)
	# Define matrix system and solve
	function f!(du, u, p, t)
	    #Q, D, alpha = p
	    du .= α' .* (Δ * Q * u) #+ ((-v_u' .* (divx * Q * u) + -v_v' .* (divy * Q * u)) ) + γ .* u
	    ###   #Diffusion         #Advection in x              #Advection in y            #Source
	end
	
	prob = ODEProblem(f!, u0, (t0, t1))
	alg = Euler()
	return prob, alg
end

function simulate(prob, alg)
  sol2D = solve(prob, alg, dt=0.005)
end

function benchmark()
  res = Dict{String, Dict{String, Real}}()
  
  #precomp run
  nesting = 2
  res["nest_pre_comp"] = Dict{String, Real}()
  
  gen_sys_val = @timed generate_sys(2^nesting)
  res["nest_pre_comp"]["generators"] = gen_sys_val[2]
  dx, dy, x, y, α, t0, t1, u0 = gen_sys_val[1]
  
  comb_val = @timed combine_sys(dx, dy, x, y)
  res["nest_pre_comp"]["construction"] = comb_val[2]
  Δ, Q = comb_val[1]
  
  init_val = @timed initialize(Δ, Q, α, t0, t1, u0)
  prob, alg = init_val[1]
  res["nest_pre_comp"]["init_sys"] = init_val[2] 
  
  sim_val = @timed simulate(prob, alg)
  res["nest_pre_comp"]["solving"] = sim_val[2]
  
  println(res["nest_pre_comp"]["generators"])
  println(res["nest_pre_comp"]["construction"])
  println(res["nest_pre_comp"]["init_sys"])
  println(res["nest_pre_comp"]["solving"])
  println("Finished nesting of pre_comp")
  
  for nesting in 2:6
    res["nest_$nesting"] = Dict{String, Real}()
  
    gen_sys_val = @timed generate_sys(2^nesting)
    res["nest_$nesting"]["generators"] = gen_sys_val[2]
    dx, dy, x, y, α, t0, t1, u0 = gen_sys_val[1]
  
    comb_val = @timed combine_sys(dx, dy, x, y)
    res["nest_$nesting"]["construction"] = comb_val[2]
    Δ, Q = comb_val[1]
  
    init_val = @timed initialize(Δ, Q, α, t0, t1, u0)
    prob, alg = init_val[1]
    res["nest_$nesting"]["init_sys"] = init_val[2] 
  
    sim_val = @timed simulate(prob, alg)
    res["nest_$nesting"]["solving"] = sim_val[2]
    println(res["nest_$nesting"]["generators"])
    println(res["nest_$nesting"]["construction"])
    println(res["nest_$nesting"]["init_sys"])
    println(res["nest_$nesting"]["solving"])
    println("Finished nesting of $nesting")
  end
end

benchmark()
