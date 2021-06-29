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
using Plots.PlotMeasures

using LaTeXStrings

# we use the notion of functorial semantics, in the wiring diagram formalism
# that means that we define empty boxes first to define the architecture
# and then assign a behavior that will inhabit those boxes compositionally

# we first have to define our boxes, with what the inputs and outputs are,
# for example, the sensor box has two input :e and :s and one output s_prime.

s = Box(:sensor    , [:e, :s]      , [:s_prime])
c = Box(:controller, [:d, :s_prime], [:c])
d = Box(:dynamics  , [:c]          , [:s])

# a wiring diagram has (inputs, outputs) which define the ports of entire diagram
UAV = WiringDiagram([:e,:d], [:s])

# associate boxes to diagram
sensor     = add_box!(UAV, s)
controller = add_box!(UAV, c)
dynamics   = add_box!(UAV, d)

add_wires!(UAV, [
    # Net Inputs
    (input_id(UAV),1) => (sensor,2),
    (input_id(UAV),2) => (controller,2),
    # Connections
    (sensor,1) => (controller,1),
    (controller,1) => (dynamics,1),
    (dynamics,1) => (sensor,1),
    # Net Outputs
    (dynamics,1) => (output_id(UAV),1)
])

# parameters:
resp_s = 100;   # response frequency of sensor
resp_c = 100;   # response frequecy of controller
gain_θ = 0;     # gain of control input proportional to sensor output (causes a feedback loop)
v_ratio = 1;    # ratio of velocity to reference velocity

function semantics(diagram)
    # State functions:
    equation_sensor(u, x, p, t)  = [ -resp_s*(u[1] - x[1] - x[2]) ];          # x = [θ, e] -> [pitch angle, pitch offset]

    equation_control(u, x, p, t) = [ -resp_c*(u[1] + gain_θ*x[1] - x[2]) ];   # x = [Sl, d] -> [sensor output, control input]

    equation_dynamic(u, x, p, t) = [-v_ratio*0.313*u[1] +  56.7*u[2] +  v_ratio*0.232*x[1],             # α -> Angle of attack
                                    (v_ratio*-0.0139*u[1] - 0.426*u[2] + v_ratio*0.0203*x[1])*v_ratio,  # q -> Angular velocity
                                    56.7*u[2]                                                        ]; # θ -> Pitch angle

    # x = [Sc] -> Controller output
    # readout functions:  [select specific state]
    readout_sensor(u)  = [ u[1] ];  # sl
    readout_control(u) = [ u[1] ];  # sc
    readout_dynamic(u) = [ u[3] ];  # θ

    # the following assings a machine to each box in the diagram of the form (inputs, states, outputs)
    s_machine = ContinuousMachine{Float64}( 2, 1, 1, equation_sensor , readout_sensor );
    c_machine = ContinuousMachine{Float64}( 2, 1, 1, equation_control, readout_control );
    d_machine = ContinuousMachine{Float64}( 1, 3, 1, equation_dynamic, readout_dynamic );

    # output composition
    return oapply( diagram, [s_machine, c_machine, d_machine] );
end

# the following assigns the behavior that ought to inhabit the boxes and then computes the total dynamics
comp = semantics(UAV)

# initial values
x_init = [0.01, 0.05];          # Inputs: [e, d] -> [θ offset, 𝛿 control input]
u_init = [0, 0, 0, 0, 0];       # States: [sl, sc, α, q, θ]

# integration interval
t_span = (0, 20);

# we have a system model but we have no way to solve it, the following does precisely that
function solveSystem()
    return solve( ODEProblem(comp, u_init, x_init, t_span), alg_hints=[:stiff] );
end

# the following plots the results of the simulation run
function plotSol(sol)
    fsize = 28
    # display solution
    plot( sol.t,                                                     # x values
          [ sol[1,:], sol[2,:], sol[3,:], 1e2*sol[4,:], sol[5,:] ],  # y values [q is amplified]
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
          bottom_margin = 15Plots.mm );
end

# plots the behavior of the system based
# on the wiring diagram definition
simulation = plotSol( solveSystem() )

display(simulation) # this can be removed if not running from the command line
