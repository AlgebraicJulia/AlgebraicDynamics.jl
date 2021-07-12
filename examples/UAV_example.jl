# authors: Georgios Bakirtzis (bakirtzis.net)
#          Raul Gonzalez Garcia (raulg@iastate.edu)
#
# the following example is a mechanization from
# 1. compositional cyber-physical systems modeling (http://dx.doi.org/10.4204/EPTCS.333.9)
# 2. categorical semantics of cyber-physical systems theory (https://doi.org/10.1145/3461669)
# please make sure to consult those to understand what is going on below

using AlgebraicDynamics
using AlgebraicDynamics.DWDDynam

using Catlab.WiringDiagrams
using Catlab.Graphics
using Catlab.Graphics.Graphviz

using DifferentialEquations

using Plots

using LabelledArrays

# we use the notion of functorial semantics, in the wiring diagram formalism
# that means that we define empty boxes first to define the architecture
# and then assign a behavior that will inhabit those boxes compositionally

# we first have to define our boxes, with what the inputs and outputs are,
# for example, the sensor box has two input :e and :s and one output s_prime.

s = Box(:sensor, [:s, :e], [:s′])
c = Box(:controller, [:d, :s′], [:c])
d = Box(:dynamics, [:c], [:s])

# a wiring diagram has (inputs, outputs) which define the ports of entire diagram
UAV = WiringDiagram([:e,:d], [:s])

sensor     = add_box!(UAV, s)
controller = add_box!(UAV, c)
dynamics   = add_box!(UAV, d)

add_wires!(UAV, [
    # net inputs
    (input_id(UAV), 1) => (sensor, 2),
    (input_id(UAV), 2) => (controller, 2),

    # connections
    (sensor, 1) => (controller, 1),
    (controller, 1) => (dynamics, 1),
    (dynamics, 1) => (sensor, 1),

    # net output
    (dynamics, 1) => (output_id(UAV), 1)
])



function 𝗟(𝐖)
    𝐿(u, x, p, t) = LVector( sc = -p.𝓐l * (u[1] - x[1] - x[2]) );
    𝐶(u, x, p, t) = LVector( sl = -p.𝓐c * (u[1] + p.𝓑c*x[1] - x[2]) );
    𝐷(u, x, p, t) = LVector( α = -0.313*u[1] +  56.7*u[2] +  0.232*x[1],
                             q = -0.0139*u[1] - 0.426*u[2] + 0.0203*x[1],
                             θ =  56.7*u[2]);

    u_𝐿(u) = [ u[1] ];  # outputs sl
    u_𝐶(u) = [ u[1] ];  # outputs sc
    u_𝐷(u) = [ u[3] ];  # outputs θ

    return oapply(𝐖,
                  Dict(:sensor     => ContinuousMachine{Float64}(2, 1, 1, 𝐿, u_𝐿),
                       :controller => ContinuousMachine{Float64}(2, 1, 1, 𝐶, u_𝐶),
                       :dynamics   => ContinuousMachine{Float64}(1, 3, 1, 𝐷, u_𝐷)));
end

# we then have to assign the behavior that ought to inhabit the boxes
𝑢ᵤₐᵥ = 𝗟(UAV)

# initial values
xₒ = LVector( e = 0.01,  # [e, d] -> [θ offset, 𝛿 control input]
              d = 0.05);

uₒ = LVector( sl = 0,
              sc = 0,
              α = 0,
              q = 0,
              θ = 0);

# integration interval
t = (0, 20);

parameters = (𝓐l = 100,  # decay constant of sensor
              𝓐c = 100,  # decay constant of controller
              𝓑c = 0);   # ratio of velocity to reference velocity

# then compute the total dynamics
solution = solve(ODEProblem(𝑢ᵤₐᵥ, uₒ, xₒ, t, parameters), alg_hints=[:stiff]);

# plot the behavior of the system based on the wiring diagram definition
fsize = 28
simulation = plot( solution.t,        # x values
                   [solution[1,:],    # y values [q is amplified]
                    solution[2,:],
                    solution[3,:],
                    solution[4,:] * 1e2,
                    solution[5,:]],

                   # labels
                   label  = ["sl" "sc" "α" "q" "θ"],
                   xlabel = "Time parameter",
                   ylabel = "Response",
                   title  = "Aircraft pitch behaviour",
                   size   = (2000,1400),

                   # font size
                   legendfontsize = fsize,
                   xtickfontsize  = fsize,
                   ytickfontsize  = fsize,
                   xguidefontsize = fsize,
                   yguidefontsize = fsize,
                   titlefontsize  = fsize,
                   lw = 4,

                   # margins
                   left_margin   = 15Plots.mm,
                   bottom_margin = 15Plots.mm);

display(simulation) # this can be removed if not running from the command line
