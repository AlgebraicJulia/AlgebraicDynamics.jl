module DWDDynam

using Catlab.Theories
using Catlab.WiringDiagrams.DirectedWiringDiagrams
using Catlab.CategoricalAlgebra
using Catlab.CategoricalAlgebra.FinSets
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

The readout function may depend on the state, parameters, and time, so it must be of the form ``r(u,p,t)``.
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

The readout function may depend on the state, parameters, and time step, so it must be of the form ``r(u_n,p,n)``.
"""
struct DiscreteMachine{T} <: AbstractMachine{T}
    ninputs::Int
    nstates::Int
    noutputs::Int
    dynamics::Function
    readout::Function
end

ContinuousMachine{T}(ninputs::Int, nstates::Int, dynamics) where T  = 
    ContinuousMachine{T}(ninputs, nstates, nstates, dynamics, (u,p,t) -> u)
DiscreteMachine{T}(ninputs::Int, nstates::Int, dynamics) where T  = 
    DiscreteMachine{T}(ninputs, nstates, nstates, dynamics, (u,p,t) -> u)
  
show(io::IO, vf::ContinuousMachine) = print("ContinuousMachine(ℝ^$(vf.nstates) × ℝ^$(vf.ninputs) → ℝ^$(vf.nstates))")
show(io::IO, vf::DiscreteMachine) = print("DiscreteMachine(ℝ^$(vf.nstates) × ℝ^$(vf.ninputs) → ℝ^$(vf.nstates))")
eltype(m::AbstractMachine{T}) where T = T

nstates(f::AbstractMachine) = f.nstates
ninputs(f::AbstractMachine) = f.ninputs
noutputs(f::AbstractMachine) = f.noutputs
readout(f::AbstractMachine, u::AbstractVector, p = nothing, t = 0) = f.readout(u, p, t)
readout(f::AbstractMachine, u::FinDomFunction, args...) = readout(f, collect(u), args...)

"""    eval_dynamics(m::AbstractMachine, u::AbstractVector, xs:AbstractVector, p, t)

Evaluates the dynamics of the machine `m` at state `u`, parameters `p`, and time `t`. The exogenous variables are set by `xs` which may either be a collection of functions ``x(t)`` or a collection of constant values. 

The length of `xs` must equal the number of inputs to `m`.
"""
eval_dynamics(f::AbstractMachine, u::AbstractVector, xs::AbstractVector, p, t::Real) = begin
    ninputs(f) == length(xs) || error("$xs must have length $(ninputs(f)) to set the exogenous variables.")
    f.dynamics(collect(u), xs, p, t)
end
eval_dynamics(f::AbstractMachine, u::T, xs::FinDomFunction, p, t) where T <: Union{AbstractVector,FinDomFunction} =
    eval_dynamics(f, collect(u), collect(xs), p, t)

eval_dynamics(f::AbstractMachine, u::AbstractVector, xs::AbstractVector{T}, p, t::Real) where T <: Function =
    eval_dynamics(f, u, [x(t) for x in xs], p, t)

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


destruct(C::Colimit, xs::FinDomFunction) = map(1:length(C)) do i 
    compose(legs(C)[i], xs)
end
destruct(C::Colimit, xs::AbstractVector) = destruct(C, FinDomFunction(xs))

"""    oapply(d::WiringDiagram, ms::Vector)

Implements the operad algebras for directed composition of dynamical systems given a 
composition pattern (implemented by a directed wiring diagram `d`)
and primitive systems (implemented by a collection of 
machines `ms`).

Each box of the composition pattern `d` is filled by a machine with the 
appropriate type signature. Returns the composite machine.
"""
function oapply(d::WiringDiagram, ms::Vector{Machine}) where {T, Machine <: AbstractMachine{T}}
    isempty(wires(d, input_id(d), output_id(d))) || error("d is not a valid composition syntax because it has pass wires")
    nboxes(d) == length(ms)  || error("there are $nboxes(d) boxes but $length(ms) machines")
    for box in 1:nboxes(d)
        fills(ms[box], d, box) || error("$ms[box] does not fill box $box")
    end

    S = coproduct((FinSet∘nstates).(ms))
    Inputs = coproduct((FinSet∘ninputs).(ms))

    function v(u::AbstractVector, xs::AbstractVector, p, t::Real)  
        states = destruct(S, u) # a list of the states by box
        readouts = map(enumerate(ms)) do (i, m) 
            readout(m, states[i], p, t)
        end 
        readins = zeros(T, length(apex(Inputs)))

        for w in wires(d, :Wire)
            readins[legs(Inputs)[w.target.box](w.target.port)] += readouts[w.source.box][w.source.port]
        end
        for w in wires(d, :InWire)
            readins[legs(Inputs)[w.target.box](w.target.port)] += xs[w.source.port]
        end

        reduce(vcat, map(enumerate(destruct(Inputs, readins))) do (i,x)
            eval_dynamics(ms[i], states[i], x, p, t)
        end)
    end

    function r(u::AbstractVector, p, t)
        states = destruct(S, u)
        readouts = map(enumerate(ms)) do (i, m)
            readout(m, states[i], p, t)
        end 
        
        outs = zeros(T, length(output_ports(d)))

        for w in wires(d, :OutWire)
            outs[w.target.port] += readouts[w.source.box][w.source.port]
        end

        return outs
    end

    return Machine(length(input_ports(d)), length(apex(S)),
                   length(output_ports(d)), v, r)
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
function oapply(d::WiringDiagram, ms::AbstractDict)
    oapply(d, [ms[box.value] for box in boxes(d)])
end

end #module
