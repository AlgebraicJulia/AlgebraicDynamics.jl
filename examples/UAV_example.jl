# authors: Georgios Bakirtzis (bakirtzis.net)
#          Raul Gonzalez Garcia (raulg@iastate.edu)

# the following example is a mechanization from:
# 1. Compositional Cyber-Physical Systems Modeling (https://arxiv.org/abs/2101.10484)
# 2. Categorical Semantics of Cyber-Physical Systems Theory (https://arxiv.org/abs/2010.08003)

using AlgebraicDynamics
using AlgebraicDynamics.DWDDynam

using Catlab.WiringDiagrams
using Catlab.Graphics
using Catlab.Graphics.Graphviz

using DifferentialEquations

using Plots

# we use the notion of functorial semantics, in the wiring diagram formalism
# that means that we define empty boxes first to define the architecture
# and then assign a behavior that will inhabit those boxes compositionally

# we first have to define our boxes, with what the inputs and outputs are,
# for example, the sensor box has two input :e and :s and one output s_prime.
s = Box(:sensor    , [:s, :e]      , [:s′])
c = Box(:controller, [:d, :s′]     , [:c])
d = Box(:dynamics  , [:c]          , [:s])

# a wiring diagram has (inputs, outputs) which define the ports of entire diagram
UAV = WiringDiagram([:e,:d], [:s])

# associate boxes to diagram
sensor     = add_box!(UAV, s)
controller = add_box!(UAV, c)
dynamics   = add_box!(UAV, d)

add_wires!(UAV, [
    # net inputs
    (input_id(UAV),1) => (sensor,2),
    (input_id(UAV),2) => (controller,2),

    # connections
    (sensor,1) => (controller,1),
    (controller,1) => (dynamics,1),
    (dynamics,1) => (sensor,1),

    # net outputs
    (dynamics,1) => (output_id(UAV),1)
])

function systemBehavior(diagram)
    # state functions:
    equation_sensor(u, x, p, t)  = [ -p.λs*(u[1] - x[1] - x[2]) ];        # x = [θ, e] -> [Pitch angle, pitch offset]

    equation_control(u, x, p, t) = [ -p.λc*(u[1] + p.kθ*x[1] - x[2]) ];   # x = [Sl, d] -> [sensor output, control input]

    equation_dynamic(u, x, p, t) = [ -0.313*p.v*u[1] +  56.7*u[2] +  0.232*p.v*x[1]      ,    # α -> Angle of attack
                                     ( -0.0139*p.v*u[1] - 0.426*u[2] + 0.0203*p.v*x[1] )*p.v,    # q -> Angular velocity
                                     56.7*u[2]                        ];   # θ -> Pitch angle
    # x = [Sc] -> Controller output
    # readout functions:  [select specific state]
    readout_sensor(u)  = [ u[1] ];  # sl
    readout_control(u) = [ u[1] ];  # sc
    readout_dynamic(u) = [ u[3] ];  # θ

    # machines to inhavit each box in diagram of the form (inputs, states, outputs)
    s_machine = ContinuousMachine{Float64}(2, 1, 1, equation_sensor , readout_sensor);
    c_machine = ContinuousMachine{Float64}(2, 1, 1, equation_control, readout_control);
    d_machine = ContinuousMachine{Float64}(1, 3, 1, equation_dynamic, readout_dynamic);

    # Output composition
    return oapply(diagram,
                   Dict(:sensor     => s_machine,
                        :controller => c_machine,
                        :dynamics   => d_machine));
end

# the following assigns the behavior that ought to inhabit the boxes and then computes the total dynamics
comp = systemBehavior(UAV)

# initial values
x_init = [0.01, 0.05];          # inputs: [e, d] -> [θ offset, 𝛿 control input]
u_init = [0, 0, 0, 0, 0];       # states: [sl, sc, α, q, θ]

# integration interval
t_span = (0, 20);

# parameters:
param = (λs = 100,  # decay constant of sensor
         λc = 100,  # decay constant of controller
         kθ = 0,    # gain of control input proportional to sensor output. Causes a feedback loop.
         v  = 1);   # ratio of velocity to reference velocity

# solve the system equations
solution = solve(ODEProblem(comp, u_init, x_init, t_span, param), alg_hints=[:stiff]);

# plot the behavior of the system based
# on the wiring diagram definition
fsize = 28

simulation = plot( solution.t,                                                                         # x values
                   [ solution[1,:], solution[2,:], solution[3,:], 1e2*solution[4,:], solution[5,:] ],  # y values [q is amplified]

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
