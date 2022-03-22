using Catlab.WiringDiagrams.DirectedWiringDiagrams
using Catlab.Graphs, Catlab.CategoricalAlgebra

using LinearAlgebra
using Test
using Plots: plot

""" 

A monomial is a pair of integers  A => B representing the poly ``\mathbb{R}^By^{\mathbb{R}^A}``
"""

const Monomial = Pair{Int,Int}
tensor(ms::Vector{Monomial}, ns::Vector{Monomial}) = sum(first.(ms)) => sum(last.(ms))

positions(m::Monomial) = m.second
directions(m::Monomial) = m.first
positions(ms::Vector{Monomial}) = sum(first.(ms))


struct DILS{T}
  input_polys::Vector{Monomial}
  output_polys::Vector{Monomial}
  nstates::Int # dimension of the state space
  forward_readout::Function
  backward_readout::Function
  eval_dynamics::Function
end

DILS{T}(input_poly::Monomial, output_poly::Monomial, nstates::Int, 
        forward_readout::Function, backward_readout::Function, eval_dynamics::Function) where T = 
  DILS{T}([input_poly], [output_poly], nstates, forward_readout, backward_readout, eval_dynamics)



input_polys(d::DILS) = d.input_polys
output_polys(d::DILS) = d.output_polys
input_poly(d::DILS) = tensor(input_polys(d))
output_poly(d::DILS) = tensor(output_polys(d))

nstates(d::DILS) = d.nstates

""" forward_readout(d::DILS, u::AbstractVector, x::AbstractVector)

This is half of the map on positions for the DILS `d` given by ``Sy^S \to [p,q]``. It takes a state ``u \in \mathbb{R}^S``
and a position in the input polynomial ``x \in p(1)`` and returns a position of the output polynomial.
"""

forward_readout(d::DILS, u::AbstractVector, x::AbstractVector) =
  d.forward_readout(u, x)
  
""" backward_readout(d::DILS, u::AbstractVector, x::AbstractVector, y::AbstractVector)

This is the other half of the map on positions. It takes a state, a position on the input polynomial 
and a position on the output polynomial and returns a direction on the input polynomial.
``\sum_{s \in S} \sum_{i\in p(1)} q[f(s,i)] \rightarrow p[i]``
"""

backward_readout(d::DILS, u::AbstractVector, x::AbstractVector, y::AbstractVector) =
  d.backward_readout(u, x, y)

""" eval_dynamics(d::DILS, u::AbstractVector, x::AbstractVector, y::AbstractVector)

This is the map on directions of the DILS `d` given by ``{\mathbb{R}^S}y^{\mathbb{R}^S} \to [p,q]`. It takes a state 
``u \in \mathbb{R}^S``, a position ``x \in p(1)``, a direction ``y \in q[f(u,x)]`` (where ``f`` is the `forward_readout`)
and returns an updated state in ``\mathbb{R}^S``
"""

eval_dynamics(d::DILS, u::AbstractVector, x::AbstractVector, y::AbstractVector) =
  d.eval_dynamics(u, x, y)

isa_state(d::DILS{T}, u::AbstractVector{T}) where T =
  length(u) == nstates(d)


# A DILS based on a function
function DILS{T}(f::Function, ninput::Int, noutput::Int; monomial=true) where T
  DILS{T}(
    monomial ? 0=>ninput : [0=>1 for _ in 1:ninput],  # input poly is ninput y^0
    monomial ? 0=>noutput : [0=>1 for _ in 1:noutput], # input poly is nouput y^0
    0,          # trivial state
    (_, x) -> f(x),  #forward map
    (_, _, _) -> T[], # backward map
    (_, _, _) -> T[]  # eval dynamics
  )
end

# A DILS that outputs constant values
DILS{T}(c::AbstractVector; monomial=true) where T = DILS{T}(_ -> c, 0, length(c); monomial=monomial)

"""
Basically a (dependent) lens
"""
function DILS{T}(forward::Function, backward::Function, ninput::Int, noutput::Int) where T
  DILS{T}(
    [ninput=>ninput],
    [noutput=>noutput],
    0,                     # trivial state
    (_, x) -> forward(x),  #forward map
    (_, x, y) -> backward(x,y), # backward map
    (_, _, _) -> T[]  # eval dynamics
    )
end

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

An unactivated_neuron with n inputs is a DILS R^{n+1} y^{R^{n+1}} -> [R^n y^{R^n}, Ry^R]
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



"""
Compose two DILS in sequences

TODO: Change braiding to be any wiring pattern from d1 to d2. 
If the output poly and input poly are of the form Ay^A (i.e. same inputs and outputs) then 
you can do this by constructing an integer-valued matrix based on the wiring pattern encoding copying, merging, delete, and create. 
For the forward pass multiply the positions of the output of d1 by the matrix. 
For the backward pass, multiply the directions of the input of d2 by the transpose of the matrix.
"""

# function sequential_compose(d1::DILS{T}, d2::DILS{T}, braiding::Nothing)::DILS{T} where T 
#   sequential_compose(d1::DILS{T})
# end

input_poly(d::DILS, i::Int) = input_polys(d)[i]
output_poly(d::DILS, i::Int) = output_polys(d)[i]

function sequential_compose(d1::DILS{T}, d2::DILS{T}, weights::AbstractMatrix)::DILS{T} where T
  # check that the monomials match
  for (i,k) in pairs(weights) 
    if k > 0 
      input_poly(d2, i[1]) == output_poly(d1, i[2])
    end
  end
  d1_state = u -> u[1:d1.nstates]
  d2_state = u -> u[d1.nstates+1:end]
  d1_output = (u,x) -> forward_readout(d1, d1_state(u), x)
  d2_backward = (u,x,y) -> backward_readout(d2, d2_state(u), d1_output(u,x), y)

  # This produces the matrices that copy and add the things flowing along the wires correctly.
  forward_matrix = generalized_kroenecker(weights, map(positions, input_polys(d2)), map(positions, output_polys(d1)))
  backward_matrix = generalized_kroenecker(weights', map(directions, output_polys(d1)), map(directions, input_polys(d2)))
  return DILS{T}(
    input_polys(d1), # input monomials
    output_polys(d2), # output monomials
    nstates(d1)+nstates(d2), # number of states/dimensions
    (u,x) -> forward_readout(d2, d2_state(u), forward_matrix*d1_output(u,x)),
    (u,x,y) -> backward_readout(d1, d1_state(u), x, backward_matrix*d2_backward(u,x,y)),
    (u,x,y) -> vcat(eval_dynamics(d1, d1_state(u), x, backward_matrix'*d2_backward(u, x, y)), # updated state for d1
                    eval_dynamics(d2, d2_state(u), forward_matrix*d1_output(u,x), y))       # updated state for d2
  )
end


function generalized_kroenecker(weights::AbstractMatrix, rowdims::Vector{Int}, coldims::Vector{Int})
  
  block = map(enumerate(coldims)) do (col, coldim)
            vcat(map(enumerate(rowdims)) do (row, rowdim)
               weights[row,col] * (rowdim == coldim ? I(rowdim) : zeros(rowdim, coldim)) 
            end...)
          end
  return hcat(block...)
end

generalized_kroenecker([1 0; 1 0; 0 1], [3,3,2],[3, 2])

minus = DILS{Float64}(x->[x[1] - x[2]], 2, 1; monomial=false)
@assert length(input_polys(minus)) == 2

constant = DILS{Float64}([1,2]; monomial=false)

one_minus_two = sequential_compose(constant, minus, [1 0; 0 1])
two_minus_one = sequential_compose(constant, minus, [0 1; 1 0])

two_minus_two = sequential_compose(constant, minus, [0 1; 2 0])

forward_readout(one_minus_two, [], []) == [-1]
forward_readout(two_minus_one, [], []) == [1]
forward_readout(two_minus_two, [], []) == [0]

backward_readout(one_minus_two, [], [], []) == []

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

"""TODO TEST"""
function unit(::DILS{T}) where T
  DILS{T}(Monomial[],Monomial[], 0, id, id, id)
end

function tensor(d1::DILS{T}, d2::DILS{T}) where T
  d1_state = u -> u[1:d1.nstates]
  d2_state = u -> u[d1.nstates+1:end]
  DILS{T}(vcat(input_polys(d1), input_polys(d2)), 
          vcat(output_polys(d1), output_polys(d2)),
          d1.nstates + d2.nstates, 
          (u,x) -> vcat(forward_readout(d1, d1_state(u), x[1:positions(input_poly(d1))], forward_readout(d2, d2_state(u), x[positions(input_poly(d1))+1,end])))
  )
end

"""TODO TEST"""
tensor(ds::Vector{D}) where {T, D<:DILS{T}} = reduce(tensor, ds, unit(DILS{T}))

function oapply(d::WiringDiagram, dils::Vector{DILS})::DILS
  # check that dils[i] "fills" box i of d
  @assert length(dils) == nboxes(d)
  for (dil, box) in zip(dils, boxes(d))
    @assert length(input_ports(d, box)) == length(input_polys(dil))
    @assert length(input_ports(d, box)) == length(output_polys(dil))
  end

  layers = layerize(internal_graph(d))

  t_layers = [tensor(dils[i] for i in layer) for layer in layers]
  res = t_layers[1]
  for (i, layer) in t_layers[2:end]
    weights = map(in_ports(d, ))
    res = compose(res, layer, weights) 
  end
  return res 

d = WiringDiagram([], [:X])
b1 = add_box!(d, Box(:const, [], [:one, :two]))
b2 = add_box!(d, Box(:minus, [:x, :y], [:x_minus_y]))
add_wires!(d, Pair[
  (b1, 1) => (b2, 2),
  (b1, 2) => (b2, 1),
  (b1, 1) => (b2, 1),
  (b2, 1) => (output_id(d), 1)
])

g = internal_graph(d)
layers = layerize(internal_graph(d))

dils = [constant, minus]

for (dil, box) in zip(dils, 1:nboxes(d))
  @assert length(input_ports(d, box)) == length(input_polys(dil)) 
  @assert length(output_ports(d, box)) == length(output_polys(dil))
end


oapply(d, dils)

"""
Partition vertices of a graph by dependency. Get ordered layers.
Can be used for composing morphisms in a SMC.
"""
function layerize(G::AbstractGraph)::Vector{Vector{Int}}
  seen = Set{Int}()
  layers = Vector{Int}[]
  while length(seen) < nv(G)
    next_layer = Int[]
    for v in filter(x->x∉ seen, parts(G, :V))
      if G[incident(G, v, :tgt) , :src] ⊆ seen
        push!(next_layer, v)
      end
    end
    push!(layers, next_layer)
    union!(seen, next_layer)
  end
  return layers
end


