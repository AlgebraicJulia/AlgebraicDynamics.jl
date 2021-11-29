module DWDDynam

using Catlab.Theories
using Catlab.WiringDiagrams.DirectedWiringDiagrams
using Catlab.CategoricalAlgebra
using Catlab.CategoricalAlgebra.FinSets
import Catlab.WiringDiagrams: oapply, input_ports, output_ports

import ..UWDDynam: nstates, eval_dynamics, euler_approx, AbstractInterface, trajectory

using OrdinaryDiffEq, DelayDiffEq
import OrdinaryDiffEq: ODEProblem, DiscreteProblem
import DelayDiffEq: DDEProblem
using Plots

export AbstractMachine, ContinuousMachine, DiscreteMachine, DelayMachine,
nstates, ninputs, noutputs, eval_dynamics, readout, euler_approx

using Base.Iterators
import Base: show, eltype, zero

### Interface
abstract type AbstractDirectedInterface{T} <: AbstractInterface{T} end

struct DirectedInterface{T} <: AbstractDirectedInterface{T}
    input_ports::Vector
    output_ports::Vector 
end
DirectedInterface{T}(ninputs::Int, noutputs::Int) where T = 
    DirectedInterface{T}(1:ninputs, 1:noutputs)

struct DirectedVectorInterface{T,N} <: AbstractDirectedInterface{T} 
    input_ports::Vector 
    output_ports::Vector
end
DirectedVectorInterface{T,N}(ninputs::Int, noutputs::Int) where {T,N} = 
    DirectedVectorInterface{T,N}(1:ninputs, 1:noutputs)

input_ports(interface::AbstractDirectedInterface) = interface.input_ports
output_ports(interface::AbstractDirectedInterface) = interface.output_ports
ninputs(interface::AbstractDirectedInterface) = length(input_ports(interface))
noutputs(interface::AbstractDirectedInterface) = length(output_ports(interface))

ndims(::DirectedVectorInterface{T, N}) where {T,N} = N


zero(::Type{I}) where {T, I<:AbstractDirectedInterface{T}} = zero(T)
zero(::Type{DirectedVectorInterface{T,N}}) where {T,N} = zeros(T,N)

### Dynamics
abstract type AbstractDirectedSystem{T} end

struct ContinuousDirectedSystem{T} <: AbstractDirectedSystem{T} 
    nstates::Int 
    dynamics::Function 
    readout::Function 
end

struct DiscreteDirectedSystem{T} <: AbstractDirectedSystem{T} 
    nstates::Int 
    dynamics::Function 
    readout::Function 
end

struct DelayDirectedSystem{T} <: AbstractDirectedSystem{T}
    nstates::Int
    dynamics::Function 
    readout::Function 
end

nstates(dynam::AbstractDirectedSystem) = dynam.nstates
dynamics(dynam::AbstractDirectedSystem) = dynam.dynamics 
readout(dynam::AbstractDirectedSystem) = dynam.readout


""" 

A directed open dynamical system operating on information fo type `T`.
A machine  `m` has type signature  (`m.ninputs`, `m.outputs`).
"""
abstract type AbstractMachine{T, InterfaceType, SystemType} end

interface(m::AbstractMachine) = m.interface 
system(m::AbstractMachine) = m.system

input_ports(m::AbstractMachine) = input_ports(interface(m))
output_ports(m::AbstractMachine) = output_ports(interface(m))
ninputs(m::AbstractMachine) = ninputs(interface(m))
noutputs(m::AbstractMachine) = noutputs(interface(m))
nstates(m::AbstractMachine) = nstates(system(m))
dynamics(m::AbstractMachine) = dynamics(system(m))
readout(m::AbstractMachine) = readout(system(m))



struct Machine{T,I,S} <: AbstractMachine{T,I,S}
    interface::I 
    system::S
end


"""    ContinuousMachine{T}(ninputs, nstates, noutputs, f, r)

An directed open continuous system. The dynamics function `f` defines an ODE ``\\dot u(t) = f(u(t),x(t),p,t)`` where ``u`` is the state and ``x`` captures the exogenous variables.

The readout function may depend on the state, parameters, and time, so it must be of the form ``r(u,p,t)``.
"""
const ContinuousMachine{T,I} = Machine{T, I, ContinuousDirectedSystem{T}}

ContinuousMachine{T}(interface::I, system::ContinuousDirectedSystem{T}) where {T, I <: AbstractDirectedInterface} = 
    ContinuousMachine{T, I}(interface, system)

ContinuousMachine{T}(ninputs, nstates, noutputs, dynamics, readout) where T = 
    ContinuousMachine{T}(DirectedInterface{T}(ninputs, noutputs), ContinuousDirectedSystem{T}(nstates, dynamics, readout))

ContinuousMachine{T}(ninputs::Int, nstates::Int, dynamics) where T  = 
    ContinuousMachine{T}(ninputs, nstates, nstates, dynamics, (u,p,t) -> u)

ContinuousMachine{T,I}(ninputs, nstates, noutputs, dynamics, readout) where {T,I <: AbstractDirectedInterface} = 
    ContinuousMachine{T,I}(I(ninputs, noutputs), ContinuousDirectedSystem{T}(nstates, dynamics, readout))

ContinuousMachine{T, N}(ninputs, nstates, noutputs, dynhamics, readout) where {T,N} = 
    ContinuousMachine{T, DirectedVectorInterface{T, N}}(ninputs, nstates, noutputs, dynhamics, readout)

ContinuousMachine{T,I}(ninputs::Int, nstates::Int, dynamics) where {T,I}  = 
    ContinuousMachine{T,I}(ninputs, nstates, nstates, dynamics, (u,p,t) -> u)


"""    DelayMachine{T}(ninputs, nstates, noutputs, f, r)

A delay open continuous system. The dynamics function `f` defines an ODE ``\\dot u(t) = f(u(t), x(t), h(p,t), p, t)`` where 
``u`` is the states, ``x`` captures the exogenous variables, and ``h`` is a history function 

The readout function may depend on the state, history, parameters, and time, so it has a signature ``r(u,h,p,t)``.
"""
const DelayMachine{T,I} = Machine{T, I, DelayDirectedSystem{T}}

DelayMachine{T}(interface::I, system::DelayDirectedSystem{T}) where {T, I<:AbstractDirectedInterface} = 
    DelayMachine{T, I}(interface, system)

DelayMachine{T}(ninputs, nstates, noutputs, dynamics, readout) where T = 
    DelayMachine{T}(DirectedInterface{T}(ninputs, noutputs), DelayDirectedSystem{T}(nstates, dynamics, readout))

DelayMachine{T}(ninputs::Int, nstates::Int, dynamics) where T  = 
    DelayMachine{T}(ninputs, nstates, nstates, dynamics, (u,p,t) -> u)

DelayMachine{T,I}(ninputs, nstates, noutputs, dynamics, readout) where {T,I<:AbstractDirectedInterface} = 
    DelayMachine{T,I}(I(ninputs, noutputs), DelayDirectedSystem{T}(nstates, dynamics, readout))

DelayMachine{T, N}(ninputs, nstates, noutputs, dynhamics, readout) where {T,N} = 
    DelayMachine{T, DirectedVectorInterface{T, N}}(ninputs, nstates, noutputs, dynhamics, readout)

DelayMachine{T,I}(ninputs::Int, nstates::Int, dynamics) where {T,I}  = 
    DelayMachine{T,I}(ninputs, nstates, nstates, dynamics, (u,h,p,t) -> u)


"""    DiscreteMachine{T}(ninputs, nstates, noutputs, f, r)

A directed open discrete dynamical system. The dynamics function `f` defines a discrete update rule ``u_{n+1} = f(u_n, x_n, p, t)`` where ``u_n`` is the state and ``x_n`` is the value of the exogenous variables at the ``n``th time step.

The readout function may depend on the state, parameters, and time step, so it must be of the form ``r(u_n,p,n)``.
"""
const DiscreteMachine{T,I} = Machine{T, I, DiscreteDirectedSystem{T}}

DiscreteMachine{T}(interface::I, system::DiscreteDirectedSystem{T}) where {T, I<:AbstractDirectedInterface} = 
    DiscreteMachine{T, I}(interface, system)

DiscreteMachine{T}(ninputs, nstates, noutputs, dynamics, readout) where T = 
    DiscreteMachine{T}(DirectedInterface{T}(ninputs, noutputs), DiscreteDirectedSystem{T}(nstates, dynamics, readout))

DiscreteMachine{T}(ninputs::Int, nstates::Int, dynamics) where T  = 
    DiscreteMachine{T}(ninputs, nstates, nstates, dynamics, (u,p,t) -> u)

DiscreteMachine{T,I}(ninputs, nstates, noutputs, dynamics, readout) where {T,I<:AbstractDirectedInterface} = 
    DiscreteMachine{T,I}(I(ninputs, noutputs), DiscreteDirectedSystem{T}(nstates, dynamics, readout))

DiscreteMachine{T, N}(ninputs, nstates, noutputs, dynhamics, readout) where {T,N} = 
    DiscreteMachine{T, VectorInterface{T, N}}(ninputs, nstates, noutputs, dynhamics, readout)

DiscreteMachine{T,I}(ninputs::Int, nstates::Int, dynamics) where {T,I}  = 
    DiscreteMachine{T,I}(ninputs, nstates, nstates, dynamics, (u,p,t) -> u)

  
show(io::IO, vf::ContinuousMachine) = print("ContinuousMachine(ℝ^$(nstates(vf)) × ℝ^$(ninputs(vf)) → ℝ^$(nstates(vf)))")
show(io::IO, vf::DelayMachine) = print("DelayMachine(ℝ^$(nstates(vf)) × ℝ^$(ninputs(vf)) → ℝ^$(nstates(vf)))")
show(io::IO, vf::DiscreteMachine) = print("DiscreteMachine(ℝ^$(nstates(vf)) × ℝ^$(ninputs(vf)) → ℝ^$(nstates(vf)))")

eltype(::AbstractMachine{T}) where T = T


readout(f::DelayMachine, u::AbstractVector, h = nothing, p = nothing, t = 0) = readout(f)(u, h, p, t)
readout(f::AbstractMachine, u::AbstractVector, p = nothing, t = 0) = readout(f)(u, p, t)
readout(f::AbstractMachine, u::FinDomFunction, args...) = readout(f, collect(u), args...)

"""    eval_dynamics(m::AbstractMachine, u::AbstractVector, xs:AbstractVector, p, t)

Evaluates the dynamics of the machine `m` at state `u`, parameters `p`, and time `t`. The exogenous variables are set by `xs` which may either be a collection of functions ``x(t)`` or a collection of constant values. 

The length of `xs` must equal the number of inputs to `m`.
"""
eval_dynamics(f::DelayMachine, u, xs, h, p=nothing, t=0) = begin
    ninputs(f) == length(xs) || error("$xs must have length $(ninputs(f)) to set the exogenous variables.")
    dynamics(f)(collect(u), collect(xs), h, p, t)
end

eval_dynamics(f::AbstractMachine, u, xs, p=nothing, t=0) = begin
    ninputs(f) == length(xs) || error("$xs must have length $(ninputs(f)) to set the exogenous variables.")
    dynamics(f)(collect(u), collect(xs), p, t)
end

# eval_dynamics(f::AbstractMachine, u::S, xs::T, args...) where {S,T <: Union{FinDomFunction, AbstractVector}} =
#     eval_dynamics(f, collect(u), collect(xs), args...)

eval_dynamics(f::AbstractMachine, u::AbstractVector, xs::AbstractVector{T}, p=nothing, t=0) where T <: Function =
    eval_dynamics(f, u, [x(t) for x in xs], p, t)

"""    euler_approx(m::ContinuousMachine, h)

Transforms a continuous machine `m` into a discrete machine via Euler's method with step size `h`. If the dynamics of `m` is given by ``\\dot{u}(t) = f(u(t),x(t),p,t)`` the the dynamics of the new discrete system is given by the update rule ``u_{n+1} = u_n + h f(u_n, x_n, p, t)``.
"""
euler_approx(f::ContinuousMachine{T}, h::Float64) where T = DiscreteMachine{T}(
    ninputs(f), nstates(f), noutputs(f), 
    (u, x, p, t) -> u + h*eval_dynamics(f, u, x, p, t),
    readout(f)
)

"""    euler_approx(m::ContinuousMachine)

Transforms a continuous machine `m` into a discrete machine via Euler's method where the step size is introduced as a new parameter, the last in the list of parameters.
"""
euler_approx(f::ContinuousMachine{T}) where T = DiscreteMachine{T}(
    ninputs(f), nstates(f), noutputs(f), 
    (u, x, p, t) -> u + p[end]*eval_dynamics(f, u, x, p[1:end-1], t),
    readout(f)
)
euler_approx(fs::Vector{M}, args...) where {M<:ContinuousMachine} = 
    map(f->euler_approx(f,args...), fs)

euler_approx(fs::AbstractDict{S, M}, args...) where {S, M<:ContinuousMachine} = 
    Dict(name => euler_approx(f, args...) for (name, f) in fs)


# Integration with ODEProblem in OrdinaryDiffEq.jl

"""    ODEProblem(m::ContinuousMachine, xs::Vector, u0::Vector, tspan, p=nothing; kwargs...)

Constructs an ODEProblem from the vector field defined by `(u,p,t) -> m.dynamics(u, x, p, t)`. The exogenous variables are determined by `xs`.
"""
ODEProblem(m::ContinuousMachine{T}, u0::AbstractVector, xs::AbstractVector, tspan, p=nothing; kwargs...)  where T= 
    ODEProblem((u,p,t) -> eval_dynamics(m, u, xs, p, t), u0, tspan, p; kwargs...)
  
ODEProblem(m::ContinuousMachine{T}, u0::AbstractVector, x::Union{T, Function}, tspan, p=nothing; kwargs...) where T= 
    ODEProblem(m, u0, collect(repeated(x, ninputs(m))), tspan, p; kwargs...)

ODEProblem(m::ContinuousMachine{T}, u0::AbstractVector, tspan, p=nothing; kwargs...) where T = 
    ODEProblem(m, u0, T[], tspan, p; kwargs...)

"""    DDEProblem(m::DelayMachine, u0::Vector, xs::Vector, h::Function, tspan, p = nothing; kwargs...)
"""
DDEProblem(m::DelayMachine, u0::AbstractVector, xs::AbstractVector, hist, tspan, params=nothing; kwargs...) = 
    DDEProblem((u,h,p,t) -> eval_dynamics(m, u, xs, h, p, t), u0, hist, tspan, params; kwargs...)


"""    DiscreteProblem(m::DiscreteMachine, xs::Vector, u0::Vector, tspan, p=nothing; kwargs...)

Constructs an DiscreteDynamicalSystem from the equation of motion defined by 
`(u,p,t) -> m.dynamics(u, x, p, t)`. The exogenous variables are determined by `xs`. Pass `nothing` in place of `p` if your system does not have parameters.
"""
DiscreteProblem(m::DiscreteMachine, u0::AbstractVector, xs::AbstractVector, tspan, p; kwargs...) = 
    DiscreteProblem((u,p,t) -> eval_dynamics(m, u, xs, p, t), u0, tspan, p; kwargs...)

DiscreteProblem(m::DiscreteMachine, u0::AbstractVector, x, tspan, p; kwargs...) = 
  DiscreteProblem(m, u0, collect(repeated(x, ninputs(m))), tspan, p; kwargs...)

DiscreteProblem(m::DiscreteMachine{T}, u0, tspan, p; kwargs...) where T = 
    DiscreteProblem(m, u0, T[], tspan, p; kwargs...)

"""    trajectory(m::DiscreteMachine, u0::AbstractVector, xs::AbstractVector, p, nsteps::Int; dt::Int = 1)

Evolves the machine `m` for `nsteps` times with step size `dt`, initial condition `u0`, and parameters `p`. Any inputs to `m` are determied by `xs`. If `m` has no inputs then you can omit `xs`.
"""
function trajectory(m::DiscreteMachine, u0::AbstractVector, xs,  p, T::Int; dt::Int= 1) 
  prob = DiscreteProblem(m, u0, xs, (0, T), p)
  sol = solve(prob, FunctionMap(); dt = dt)
  return sol.u
end
    

### Plotting backend
@recipe function f(sol, m::AbstractMachine, p=nothing)
    labels = (String ∘ Symbol).(output_ports(m))
    label --> reshape(labels, 1, length(labels))
    vars --> map(1:noutputs(m)) do i
        ((t, args...) -> (t, readout(m)(collect(args), p, t)[i]), 0:nstates(m)...)
    end
    sol
end


"""    oapply(d::WiringDiagram, ms::Vector)

Implements the operad algebras for directed composition of dynamical systems given a 
composition pattern (implemented by a directed wiring diagram `d`)
and primitive systems (implemented by a collection of 
machines `ms`).

Each box of the composition pattern `d` is filled by a machine with the 
appropriate type signature. Returns the composite machine.
"""
function oapply(d::WiringDiagram, ms::Vector{M}) where {M<:AbstractMachine}
    isempty(wires(d, input_id(d), output_id(d))) || error("d is not a valid composition syntax because it has pass wires")
    nboxes(d) == length(ms)  || error("there are $nboxes(d) boxes but $length(ms) machines")
    for box in 1:nboxes(d)
        fills(ms[box], d, box) || error("$ms[box] does not fill box $box")
    end

    S = coproduct((FinSet∘nstates).(ms))

    return M(input_ports(d), 
        length(apex(S)),
        output_ports(d), 
        induced_dynamics(d, ms, S), 
        induced_readout(d, ms, S))
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



### Helper functions for `oapply`

function induced_dynamics(d::WiringDiagram, ms::Vector{M}, S) where {T,I, M<:AbstractMachine{T,I}}

    function v(u::AbstractVector, xs::AbstractVector, p, t::Real)  
        states = destruct(S, u) # a list of the states by box
        readouts = get_readouts(ms, states, p, t)
      
        reduce(vcat, map(1:nboxes(d)) do i
          inputs = map(1:length(input_ports(d,i))) do port
            sum(map(in_wires(d,i,port)) do w
              ys = w.source.box == input_id(d) ? xs : readouts[w.source.box]
              ys[w.source.port]
            end; init = zero(I))
          end
          eval_dynamics(ms[i], states[i], inputs, p, t)
        end)
    end
end

function induced_dynamics(d::WiringDiagram, ms::Vector{M}, S) where {T,I, M<:DelayMachine{T,I}}

    function v(u::AbstractVector, xs::AbstractVector, h, p, t::Real) 
        states = destruct(S, u) # a list of the states by box
        hists = destruct(S, h)
        readouts = get_readouts(ms, states, hists, p, t)

        reduce(vcat, map(1:nboxes(d)) do i

          inputs = map(1:length(input_ports(d,i))) do port
            sum(map(in_wires(d,i,port)) do w
              ys = w.source.box == input_id(d) ? xs : readouts[w.source.box]
              ys[w.source.port]
            end; init = zero(I))
          end

          eval_dynamics(ms[i], states[i],  inputs, hists[i], p, t)

        end)
    end
end 
    
function induced_readout(d::WiringDiagram, ms::Vector{M}, S) where {T, I, M<:AbstractMachine{T,I}}
    function r(u::AbstractVector, p, t)
        states = destruct(S, u)
        readouts = get_readouts(ms, states, p, t)

        map(1:length(output_ports(d))) do p
          sum(map(in_wires(d, output_id(d), p)) do w
            readouts[w.source.box][w.source.port]
          end; init = zero(I))
        end
    end
end

function induced_readout(d::WiringDiagram, ms::Vector{M}, S) where {T, I, M<:DelayMachine{T,I}}
    function r(u::AbstractVector, h, p, t)
        states = destruct(S, u)
        hists = destruct(S, h)
        readouts = get_readouts(ms, states, hists, p, t)
        
        map(1:length(output_ports(d))) do p
          sum(map(in_wires(d, output_id(d), p)) do w
            readouts[w.source.box][w.source.port]
          end; init = zero(I))
        end
    end
end


"""    fills(m::AbstractMachine, d::WiringDiagram, b::Int)

Checks if `m` is of the correct signature to fill box `b` of the  wiring diagram `d`.
"""
function fills(m::AbstractMachine, d::WiringDiagram, b::Int)
    b <= nboxes(d) || error("Trying to fill box $b, when $d has fewer than $b boxes")
    b = box_ids(d)[b]
    return ninputs(m) == length(input_ports(d,b)) && noutputs(m) == length(output_ports(d,b))
end


destruct(C::Colimit, xs::FinDomFunction) = map(1:length(C)) do i 
    collect(compose(legs(C)[i], xs))
end
destruct(C::Colimit, xs::AbstractVector) = destruct(C, FinDomFunction(xs))

destruct(C::Colimit, h) = map(1:length(C)) do i 
    (p,t) -> destruct(C, h(p,t))[i]
end

get_readouts(ms::AbstractArray{M}, states, p, t) where {M <: AbstractMachine} = map(enumerate(ms)) do (i, m) 
    readout(m, states[i], p, t)
end 

get_readouts(ms::AbstractArray{M}, states, hists, p, t) where {M<:DelayMachine} = map(enumerate(ms)) do (i, m) 
    readout(m, states[i], hists[i], p, t)
end



end #module
