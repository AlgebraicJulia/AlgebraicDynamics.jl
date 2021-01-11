module DWDDynam

using Catlab.WiringDiagrams.DirectedWiringDiagrams
using  Catlab.CategoricalAlgebra
using  Catlab.CategoricalAlgebra.FinSets
import Catlab.WiringDiagrams: oapply

import ..UWDDynam: nstates, eval_dynamics, euler_approx

using OrdinaryDiffEq, DynamicalSystems
import OrdinaryDiffEq: ODEProblem
import DynamicalSystems: DiscreteDynamicalSystem

export AbstractMachine, ContinuousMachine, DiscreteMachine, 
nstates, ninputs, noutputs, eval_dynamics, readout, euler_approx

using Base.Iterators
import Base: show, eltype

""" 

A directed open dynamical system operating on information fo type `T`.
A machine  `m` has type signature  (`m.ninputs`, `m.outputs`).
"""
abstract type AbstractMachine{T} end


"""

An undirected open continuous system. The dynamics function `f` defines an ODE ``\\dot u(t) = f(u(t),x(t),p,t)`` where ``u`` is the state and ``x`` captures the exogenous variables.

The readout function may only depend on the state, so it must be of the form ``r(u(t))``.
"""
struct ContinuousMachine{T} <: AbstractMachine{T}
    ninputs::Int
    nstates::Int
    noutputs::Int
    dynamics::Function
    readout::Function
end

"""

A directed open discrete dynamical system. The dynamics function `f` defines a discrete update rule ``u_{n+1} = f(u_n, x_n, p, t)`` where ``u_n`` is the state and ``x_n`` is the value of the exogenous variables at the ``n``th time step.

The readout function may only depend on the state, so it must be of the form ``r(u_n)``.
"""
struct DiscreteMachine{T} <: AbstractMachine{T}
    ninputs::Int
    nstates::Int
    noutputs::Int
    dynamics::Function
    readout::Function
end
  
show(io::IO, vf::ContinuousMachine) = print("ContinuousMachine(ℝ^$(vf.nstates) × ℝ^$(vf.ninputs) → ℝ^$(vf.nstates))")
show(io::IO, vf::DiscreteMachine) = print("DiscreteMachine(ℝ^$(vf.nstates) × ℝ^$(vf.ninputs) → ℝ^$(vf.nstates))")
eltype(m::AbstractMachine{T}) where T = T

nstates(f::AbstractMachine) = f.nstates
ninputs(f::AbstractMachine) = f.ninputs
noutputs(f::AbstractMachine) = f.noutputs
readout(f::AbstractMachine, u::AbstractVector) = f.readout(u)

"""    eval_dynamics(m::AbstractMachine, u::AbstractVector, xs::AbstractVector, p, t)

Evaluates the dynamics of the machine `m` at state `u`, parameters `p`, and time `t`. The exogenous variables are set by `xs` which may either be a collection of functions ``x(t)`` or a collection of constant values. 

The length of `xs` must equal the number of inputs to `m`.
"""
eval_dynamics(f::AbstractMachine, u::AbstractVector, xs::AbstractVector{T}, p, t::Real) where T <: Function =
    eval_dynamics(f, u, [x(t) for x in xs], p, t)

eval_dynamics(f::AbstractMachine, u::AbstractVector, xs::AbstractVector, p, t::Real)  = begin
    ninputs(f) == length(xs) || error("$xs must have length $(ninputs(f)) to set the exogenous variables.")
    f.dynamics(u, xs, p, t)
end
    
eval_dynamics(f::AbstractMachine, u::AbstractVector, xs::AbstractVector) = 
    eval_dynamics(f, u, xs, nothing, 0)
eval_dynamics(f::AbstractMachine, u::AbstractVector, xs::AbstractVector, p) = 
    eval_dynamics(f, u, xs, p, 0)


"""    euler_approx(m::ContinuousMachine, h)

Transforms a continuous machine `m` into a discrete machine via Euler's method with step size `h`. If the dynamics of `m` is given by ``\\dot{u}(t) = f(u(t),x(t),p,t)`` the the dynamics of the new discrete system is given by the update rule ``u_{n+1} = u_n + h f(u_n, x_n, p, t)``.
"""
euler_approx(f::ContinuousMachine{T}, h::Float64) where T = DiscreteMachine{T}(
    ninputs(f), nstates(f), noutputs(f), 
    (u, x, p, t) -> u + h*eval_dynamics(f, u, x, p, t),
    f.readout
)

"""    euler_approx(m::ContinuousMachine)

Transforms a continuous machine `m` into a discrete machine via Euler's method where the step size is introduced as a new parameter, the last in the list of parameters.
"""
euler_approx(f::ContinuousMachine{T}) where T = DiscreteMachine{T}(
    ninputs(f), nstates(f), noutputs(f), 
    (u, x, p, t) -> u + p[end]*eval_dynamics(f, u, x, p[1:end-1], t),
    f.readout
)
euler_approx(fs::Vector{ContinuousMachine{T}}, args...) where T = 
    map(f->euler_approx(f,args...), fs)

euler_approx(fs::AbstractDict{S, ContinuousMachine{T}}, args...) where {S, T} = 
    Dict(name => euler_approx(f, args...) for (name, f) in fs)


# Integration with ODEProblem in OrdinaryDiffEq.jl

"""    ODEProblem(m::ContinuousMachine, xs::Vector, u0::Vector, tspan)

Constructs an ODEProblem from the vector field defined by `(u,p,t) -> m.dynamics(u, x, p, t)`. The exogenous variables are determined by `xs`.
"""
ODEProblem(m::ContinuousMachine, u0::AbstractVector, xs::AbstractVector, tspan::Tuple{Real, Real}, p=nothing)  = 
    ODEProblem((u,p,t) -> eval_dynamics(m, u, xs, p, t), u0, tspan, p)
  
ODEProblem(m::ContinuousMachine, u0::AbstractVector, x, tspan::Tuple{Real, Real}, p=nothing) = 
    ODEProblem(m, u0, collect(repeated(x, ninputs(m))), tspan, p)

ODEProblem(m::ContinuousMachine{T}, u0::AbstractVector, tspan::Tuple{Real, Real}, p=nothing) where T = 
    ODEProblem(m, u0, T[], tspan, p)

"""    DiscreteDynamicalSystem(m::DiscreteMachine, xs::Vector, u0::Vector, p)

Constructs an DiscreteDynamicalSystem from the equation of motion defined by 
`(u,p,t) -> m.dynamics(u, x, p, t)`. The exogenous variables are determined by `xs`. Pass `nothing` in place of `p` if your system does not have parameters.
"""
DiscreteDynamicalSystem(m::DiscreteMachine, u0::AbstractVector, x,  p; t0::Int = 0) = 
  DiscreteDynamicalSystem(m, u0, collect(repeated(x, ninputs(m))), p; t0=t0)
  
DiscreteDynamicalSystem(m::DiscreteMachine{T}, u0::AbstractVector, xs::AbstractVector, p; t0::Int = 0) where T = begin
  if nstates(m) == 1
    DiscreteDynamicalSystem1d(m, u0[1], xs, p; t0=t0)
  else
    !(T <: AbstractFloat) || error("Cannot construct a DiscreteDynamicalSystem if the type is a float")
    DiscreteDynamicalSystem(
        (u,p,t) -> SVector{nstates(m)}(eval_dynamics(m, u, xs, p, t)), 
        u0, p; t0=t0)
  end
end

DiscreteDynamicalSystem(m::DiscreteMachine, u0::Real, xs::AbstractVector, p; t0::Int = 0) = 
  DiscreteDynamicalSystem1D(m, u0, xs, p; t0=t0) 

# if the system is 1D then the state must be represented by a number NOT by a 1D array
DiscreteDynamicalSystem1d(m::DiscreteMachine{T}, u0::Real, xs::AbstractVector, p; t0::Int = 0) where T = begin
  nstates(m) == 1 || error("The machine must have exactly 1 state")
  !(T <: AbstractFloat) || error("Cannot construct a DiscreteDynamicalSystem if the type is a float")
  DiscreteDynamicalSystem(
      (u,p,t) -> eval_dynamics(m, [u], xs, p, t)[1], 
      u0, p; t0=t0)
end

DiscreteDynamicalSystem(m::DiscreteMachine, u0, p; t0::Int = 0) = 
    DiscreteDynamicalSystem(m, u0, [], p; t0 = t0)


"""    fills(m::AbstractMachine, d::WiringDiagram, b::Int)

Checks if `m` is of the correct signature to fill box `b` of the  wiring diagram `d`.
"""
function fills(m::AbstractMachine, d::WiringDiagram, b::Int)
    b <= nboxes(d) || error("Trying to fill box $b, when $d has fewer than $b boxes")
    b = box_ids(d)[b]
    return ninputs(m) == length(input_ports(d,b)) && noutputs(m) == length(output_ports(d,b))
end

colimitmap!(f::Function, output, C::Colimit, input) = begin
    for (i,x) in enumerate(input)
        y = f(i, x)
        I = legs(C)[i](1:length(y))
        # length(I) == length(y) || error("colimitmap! attempting to fill $(length(I)) slots with $(length(y)) values")
        output[I] .= y
    end
    return output
end

@inline fillreadouts!(y, d, xs, Outputs, statefun) = colimitmap!(y, Outputs, xs) do i,x
    return x.readout(statefun(i))
end

@inline fillstates!(y, d, xs, States, statefun, inputfun, p, t) = colimitmap!(y, States, xs) do i, x
    return x.dynamics(statefun(i), inputfun(i), p, t)
end

@inline fillwire(w, d, readouts, Outputs) = readouts[legs(Outputs)[w.source.box - 2](w.source.port)] # FIX - box re-indexing

fillreadins!(readins, d, readouts, Outputs, Inputs, ins) = begin
    for (i,w) in enumerate(wires(d))
        if w.target.box == output_id(d)
            continue
        elseif w.source.box == input_id(d)
            readins[legs(Inputs)[w.target.box - 2](w.target.port)] += ins[w.source.port]
        else
            readins[legs(Inputs)[w.target.box - 2](w.target.port)] += fillwire(w, d, readouts, Outputs)
        end
        
    end
    return readins
end

"""    oapply(d::WiringDiagram, ms::Vector)

Implements the operad algebras for directed composition of dynamical systems given a 
composition pattern (implemented by a directed wiring diagram `d`)
and primitive systems (implemented by a collection of 
machines `ms`).

Each box of the composition pattern `d` is filled by a machine with the 
appropriate type signature. Returns the composite machine.
"""
function oapply(d::WiringDiagram, xs::Vector{Machine}) where {T, Machine <: AbstractMachine{T}}
    isempty(wires(d, input_id(d), output_id(d))) || error("d is not a valid composition syntax because it has pass wires")
    nboxes(d) == length(xs)  || error("there are $nboxes(d) boxes but $length(xs) machines")
    for box in 1:nboxes(d)
        fills(xs[box], d, box) || error("$xs[box] does not fill box $box")
    end

    S = coproduct((FinSet∘nstates).(xs))
    Inputs = coproduct((FinSet∘ninputs).(xs))
    Outputs = coproduct((FinSet∘noutputs).(xs))
    ys = zeros(T, length(apex(S)))

    states(u::AbstractVector, b::Int) = u[legs(S)[b](1:xs[b].nstates)]

    v = (u::AbstractVector, ins::AbstractVector, p, t::Real) -> begin
        readouts = zeros(T, length(apex(Outputs)))
        readins = zeros(T, length(apex(Inputs)))

        get_states(b) = states(u,b)
        get_inputs(b) = view(readins, legs(Inputs)[b](:))
        
        fillreadouts!(readouts, d, xs, Outputs, get_states)
        fillreadins!(readins, d, readouts, Outputs, Inputs, ins)
        fillstates!(ys, d, xs, S, get_states, get_inputs, p, t)
        return ys
    end

    function readout(u::AbstractVector)
        readouts = zeros(T, length(apex(Outputs)))
        get_states(b) = states(u,b)

        fillreadouts!(readouts, d, xs, Outputs, get_states)
        r = zeros(T, length(d.output_ports))
        for w in in_wires(d, output_id(d))
            r[w.target.port] += fillwire(w, d, readouts, Outputs)
        end
        return r
    end

    return Machine(length(d.input_ports), length(apex(S)), length(d.output_ports), v, readout)
    
end

"""    oapply(d::WiringDiagram, m::AbstractMachine)

A version of `oapply` where each box of `d` is filled with the machine `m`.
"""
function oapply(d::WiringDiagram, x::AbstractMachine)
    oapply(d, collect(repeated(x, nboxes(d))))
end

"""    oapply(d::WiringDiagram, generators::Dict)

A version of `oapply` where `generators` is a dictionary mapping the name of each box to its corresponding machine. 
"""
function oapply(d::WiringDiagram, xs::AbstractDict)
    oapply(d, [xs[box.value] for box in boxes(d)])
end

end #module