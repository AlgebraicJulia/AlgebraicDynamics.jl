using Revise
using AlgebraicDynamics.DWDDILS

using Test
using LinearAlgebra
using Plots: plot

using Catlab.WiringDiagrams.DirectedWiringDiagrams



# Functional DILS
#################

minus = DILS{Float64}(x->[x[1] - x[2]], 2, 1; monomial=false)
constant = DILS{Float64}([1,2]; monomial=false)

@test length(input_polys(minus)) == 2

# Sequential composition
########################
one_minus_two = sequential_compose(constant, minus, [1 0; 0 1])
two_minus_one = sequential_compose(constant, minus, [0 1; 1 0])
two_minus_two = sequential_compose(constant, minus, [0 1; 2 0])

@test forward_readout(one_minus_two, [], []) == [-1]
@test forward_readout(two_minus_one, [], []) == [1]
@test forward_readout(two_minus_two, [], []) == [0]
@test backward_readout(one_minus_two, [], [], []) == []

# Unit
######
U = unit(DILS{Float64})
@test isempty(forward_readout(U, [], []))
@test isempty(backward_readout(U, [], [], []))

t = tensor(unit(typeof(one_minus_two)), one_minus_two);
@test nstates(t) == nstates(one_minus_two)
@test input_poly(t) == input_poly(one_minus_two)
@test output_poly(t) == output_poly(one_minus_two)
@test forward_readout(t, [], []) == [-1]
@test backward_readout(t, [], [], []) == []

# Oapply
########

d = WiringDiagram([], [:X])
b1 = add_box!(d, Box(:const, [], [:one, :two]))
b2 = add_box!(d, Box(:minus, [:x, :y], [:x_minus_y]))
add_wires!(d, Pair[
  (b1, 1) => (b2, 2),
  (b1, 2) => (b2, 1),
  (b1, 1) => (b2, 1),
  (b2, 1) => (output_id(d), 1)
])

dils = [constant, minus]
res = oapply(d, dils);
@test forward_readout(res, [], []) == [2.]

# NN
####

# A DILS with 1 -> [Ry^R,Ry^R] representing "relu" with gradient descent
relu = DILS{Float64}(
    (x) -> [max(0, x[1])],          # relu
    (x,y)-> x[1] < 0 ? [0] : y,     # gradient of relu
    1, 1
)

@test forward_readout(relu, [], [-1]) == [0]
@test forward_readout(relu, [], [1]) == [1]
@test backward_readout(relu, [], [10], [1]) == [1]
@test backward_readout(relu, [], [-10], [1]) == [0]

"""Gradient descent dynamics

An unactivated_neuron with n inputs is a DILS:
 R^{n+1} y^{R^{n+1}} -> [R^n y^{R^n}, Ry^R]
"""
function unactivated_neuron(n_inputs::Int)
  DILS{Float64}(
    [n_inputs => n_inputs],           # input
    [1 => 1],                         # outputs
    n_inputs+1,                       # state (bias is last element)
    (u, x) -> [dot(u, vcat(x,[1]))],  #forward readout
    (u, x, Δy) -> Δy .* u[1:end-1],          # backward readout - returns Δx
    (u, x, Δy) -> Δy .* vcat(x, [1])         # eval_dynamics - returns Δu
  )
end

"""Note, this tests the default sequential_compose (no matrix given)"""
activated_neuron(ninputs::Int) =
    sequential_compose(unactivated_neuron(ninputs), relu)

my_fun(x) = [max(0, x[1] + 2*x[2] + 5)]
my_fun(x,y) = my_fun([x,y])

function training_data(f::Function, n_steps::Int=1000, low::Float64=-10., high::Float64=10.)
  return map(1:n_steps) do _
    x = [rand() * (high-low) + low for _ in 1:2]
    x => f(x)
  end
end


mutable struct SingleNeuron
    weights::Vector
    bias::Vector
    d::DILS{Float64}
end


function initialize!(neuron::SingleNeuron)
  neuron.weights = rand(2)
  neuron.bias = rand(1)
end

function train!(neuron::SingleNeuron, training_data,
                α::Float64=0.01)
  ys = Float64[]
  for (x,y) in training_data
    u = vcat(neuron.weights, neuron.bias)
    y′ =forward_readout(neuron.d, u, x)
    Δy = y′.- y
    push!(ys, Δy[1])
    new_u = u - eval_dynamics(neuron.d, u, x, α*Δy)
    neuron.weights = new_u[1:2]
    neuron.bias = [new_u[3]]
  end
  return ys
end

(s::SingleNeuron)(x,y) = forward_readout(s.d, vcat(s.weights, s.bias), [x,y])


weights = rand(2)
bias = rand(1)
nn = SingleNeuron(weights, bias, activated_neuron(2))
nn(1.,1.)


data = training_data(my_fun, 10000)
initialize!(nn)
errs = train!(nn, data)

nonzero_errs = collect(filter(x->x!=(0),errs))
plot(1:length(nonzero_errs), nonzero_errs)

# plot(first.(data), last.(data)) # only if data is 1-D


# MISC
######
@test generalized_kroenecker([1 0; 1 0; 0 1], [3,3,2],[3, 2]) == [
    1 0 0 0 0;
    0 1 0 0 0;
    0 0 1 0 0;
    1 0 0 0 0;
    0 1 0 0 0;
    0 0 1 0 0;
    0 0 0 1 0;
    0 0 0 0 1
]
