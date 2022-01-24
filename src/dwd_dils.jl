using Catlab.WiringDiagrams.DirectedWiringDiagrams
using LinearAlgebra
using Test
using Plots: plot


const Monomial = Pair{Int,Int}


struct DILS{T}
  input_poly::Monomial
  output_poly::Monomial
  nstates::Int # dimension of the state space
  forward_readout::Function
  backward_readout::Function
  eval_dynamics::Function
end


input_poly(d::DILS) = d.input_poly
output_poly(d::DILS) = d.output_poly
nstates(d::DILS) = d.nstates

forward_readout(d::DILS, u::AbstractVector, x::AbstractVector) =
  d.forward_readout(u, x)
backward_readout(d::DILS, u::AbstractVector, x::AbstractVector, y::AbstractVector) =
  d.backward_readout(u, x, y)
eval_dynamics(d::DILS, u::AbstractVector, x::AbstractVector, y::AbstractVector) =
  d.eval_dynamics(u, x, y)

isa_state(d::DILS{T}, u::AbstractVector{T}) where T =
  length(u) == nstates(d)


function DILS{T}(f::Function, ninput::Int, noutput::Int) where T
  DILS{T}(
    0=>ninput,
    0=>noutput,
    0,
    (_, x) -> f(x)),  #forward map
    (_, _, _) -> T[], # backward map
    (_, _, _) -> T[]  # eval dynamics
end


"""
forward(x)::
backward(x,y)
"""
function DILS{T}(forward::Function, backward::Function, ninput::Int, noutput::Int) where T
  DILS{T}(
    ninput=>ninput,
    noutput=>noutput,
    0,
    (_, x) -> forward(x),  #forward map
    (_, x, y) -> backward(x,y), # backward map
    (_, _, _) -> T[]  # eval dynamics
    )
end

relu = DILS{Float64}(
    (x) -> [max(0, x[1])],
    (x,y)-> x[1] < 0 ? [0] : y,
    1, 1
)

@test forward_readout(relu, [], [-1]) == [0]

"""Gradient descent dynamics"""
function unactivated_neuron(n_inputs::Int)
  DILS{Float64}(
    n_inputs => n_inputs, # input
    1 => 1, # outputs
    n_inputs+1, # state (bias is last element)
    (u, x) -> [dot(u, vcat(x,[1]))],  #forward readout
    (u, x, Δy) -> Δy .* u[1:end-1],          # backward readout - returns Δx
    (u, x, Δy) -> Δy .* vcat(x, [1])         # eval_dynamics - returns Δu
  )
end


"""
Compose two DILS in sequences
"""
function sequential_compose(d1::DILS{T}, d2::DILS{T})::DILS{T} where T
  d1_state = u -> u[1:d1.nstates]
  d2_state = u -> u[d1.nstates+1:end]
  d1_output = (u,x) -> forward_readout(d1, d1_state(u), x)
  d2_backward = (u,x,y) -> backward_readout(d2, d2_state(u), d1_output(u,x), y)
  return DILS{T}(
    input_poly(d1), # input monomial
    output_poly(d2), # output monomial
    nstates(d1)+nstates(d2), # number of states/dimensions
    (u,x) -> forward_readout(d2, d2_state(u), d1_output(u,x)),
    (u,x,y) -> backward_readout(d1, d1_state(u), x, d2_backward(u,x,y)),
    (u,x,y) -> vcat(eval_dynamics(d1, d1_state(u), x, d2_backward(u, x, y)), # updated state for d1
                    eval_dynamics(d2, d2_state(u), d1_output(u,x), y))       # updated state for d2
  )
end

activated_neuron(ninputs::Int) = sequential_compose(unactivated_neuron(ninputs), relu)




mutable struct SingleNeuron
  weights::Vector
  bias::Vector
  d::DILS{Float64}
end


(s::SingleNeuron)(x,y) = forward_readout(s.d, vcat(s.weights, s.bias), [x,y])


weights = rand(2)
bias = rand(1)
nn = SingleNeuron(weights, bias, activated_neuron(2))
nn(1.,1.)


my_fun(x) = [max(0, x[1] + 2*x[2] + 5)]
my_fun(x,y) = my_fun([x,y])

function training_data(f::Function, n_steps::Int=1000, low::Float64=-10., high::Float64=10.)
  return map(1:n_steps) do _
    x = [rand() * (high-low) + low for _ in 1:2]
    x => f(x)
  end
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


data = training_data(my_fun, 10000)
initialize!(nn)
errs = train!(nn, data)

nonzero_errs = collect(filter(x->x!=(0),errs))
plot(1:length(nonzero_errs), nonzero_errs)

plot(first.(data), last.(data))


function oapply(d::WiringDiagram, dils::Vector{DILS})::DILS
end


