module DWDDynam

using Catlab

import Catlab.WiringDiagrams: oapply, input_ports, output_ports

import ..UWDDynam: AbstractInterface, nstates, eval_dynamics, euler_approx, trajectory


export AbstractMachine, ContinuousMachine, DiscreteMachine, DelayMachine, 
InstantaneousContinuousMachine, InstantaneousDiscreteMachine, InstantaneousDelayMachine, nstates, ninputs, noutputs, eval_dynamics, readout, euler_approx, 
dependency_pairs, output_ports

using Base.Iterators
import Base: show, eltype, zero

######## Interfaces #######
abstract type AbstractDirectedInterface{T} <: AbstractInterface{T} end

# DirectedInterface
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

# Instantaneous interfaces
abstract type AbstractInstantaneousDirectedInterface{T} <: AbstractDirectedInterface{T} end 

# InstantaneousDirectedInterface
struct InstantaneousDirectedInterface{T} <: AbstractInstantaneousDirectedInterface{T}
  input_ports::Vector 
  output_ports::Vector
  dependency::Span # P_in <- R -> P_out
end

InstantaneousDirectedInterface{T}(input_ports::AbstractVector, output_ports::AbstractVector, dependency_pairs::AbstractVector) where {T} = 
  InstantaneousDirectedInterface{T}(input_ports, output_ports, 
    Span(
      FinFunction(Array{Int}(last.(dependency_pairs)), length(dependency_pairs), length(input_ports)), 
      FinFunction(Array{Int}(first.(dependency_pairs)), length(dependency_pairs), length(output_ports))
    ))

InstantaneousDirectedInterface{T}(ninputs::Int, noutputs::Int, dependency) where {T} = 
  InstantaneousDirectedInterface{T}(1:ninputs, 1:noutputs, dependency)

InstantaneousDirectedInterface{T}(input_ports::AbstractVector, output_ports::AbstractVector, ::Nothing) where T = 
  InstantaneousDirectedInterface{T}(input_ports, output_ports, vcat(
    map(1:length(input_ports)) do i 
      map(1:length(output_ports)) do j 
        j => i
      end
    end...)
  )

# InstantaneousDirectedVectorInterface
struct InstantaneousDirectedVectorInterface{T,N} <: AbstractInstantaneousDirectedInterface{T}
  input_ports::Vector 
  output_ports::Vector
  dependency::Span # P_in <- R -> P_out
end

InstantaneousDirectedVectorInterface{T,N}(input_ports::AbstractVector, output_ports::AbstractVector, dependency_pairs::AbstractVector) where {T,N} = 
InstantaneousDirectedVectorInterface{T,N}(input_ports, output_ports, 
    Span(
      FinFunction(Array{Int}(last.(dependency_pairs)), length(dependency_pairs), length(input_ports)), 
      FinFunction(Array{Int}(first.(dependency_pairs)), length(dependency_pairs), length(output_ports))
    ))

InstantaneousDirectedVectorInterface{T,N}(ninputs::Int, noutputs::Int, dependency) where {T,N} = 
    InstantaneousDirectedVectorInterface{T,N}(1:ninputs, 1:noutputs, dependency)

InstantaneousDirectedVectorInterface{T,N}(input_ports::AbstractVector, output_ports::AbstractVector, ::Nothing) where {T,N} = 
  InstantaneousDirectedVectorInterface{T,N}(input_ports, output_ports, vcat(
    map(1:length(input_ports)) do i 
      map(1:length(output_ports)) do j 
        j => i
      end
    end...)
)

# get dependency
dependency(interface::AbstractInstantaneousDirectedInterface) = interface.dependency
dependency_pairs(interface::AbstractInstantaneousDirectedInterface) = map(apex(dependency(interface))) do i 
  legs(dependency(interface))[2](i) => legs(dependency(interface))[1](i)
end |> sort

# methods for directed interfaces
input_ports(interface::AbstractDirectedInterface) = interface.input_ports
output_ports(interface::AbstractDirectedInterface) = interface.output_ports
ninputs(interface::AbstractDirectedInterface) = length(input_ports(interface))
noutputs(interface::AbstractDirectedInterface) = length(output_ports(interface))

ndims(::DirectedVectorInterface{T, N}) where {T,N} = N
ndims(::InstantaneousDirectedVectorInterface{T, N}) where {T,N} = N

zero(::Type{I}) where {T, I<:AbstractDirectedInterface{T}} = zero(T)
zero(::Type{DirectedVectorInterface{T,N}}) where {T,N} = zeros(T,N)
zero(::Type{InstantaneousDirectedVectorInterface{T,N}}) where {T,N} = zeros(T,N)

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


"""    Machine{T}
    Machine{T,N}

A directed open dynamical system operating on information of type `T`.
For type arguments `{T,N}`, the system operates on arrays of type `T` and `ndims = N`.
A machine  `m` has type signature  (`m.ninputs`, `m.noutputs`).
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

dependency(m::Machine{T,I}) where {T, I<:InstantaneousDirectedInterface{T}} = dependency(interface(m))
dependency_pairs(m::Machine{T,I}) where {T, I<:InstantaneousDirectedInterface{T}} = dependency_pairs(interface(m))

"""    ContinuousMachine{T}(ninputs, nstates, noutputs, f, r)

A directed open continuous system. The dynamics function `f` defines an ODE ``\\dot u(t) = f(u(t),x(t),p,t)``, where
``u`` is the state and ``x`` captures the exogenous variables.

The readout function `r` may depend on the state, parameters and time, so it must be of the form ``r(u,p,t)``.
If it is left out, then ``r=u``.
"""
const ContinuousMachine{T,I} = Machine{T, I, ContinuousDirectedSystem{T}}
const InstantaneousContinuousMachine{T,I} = Machine{T, I, ContinuousDirectedSystem{T}}

ContinuousMachine{T}(interface::I, system::ContinuousDirectedSystem{T}) where {T, I <: AbstractDirectedInterface} =
    ContinuousMachine{T, I}(interface, system)

ContinuousMachine{T}(ninputs, nstates, noutputs, dynamics, readout) where T =
    ContinuousMachine{T}(DirectedInterface{T}(ninputs, noutputs), ContinuousDirectedSystem{T}(nstates, dynamics, readout))

ContinuousMachine{T,I}(ninputs, nstates, noutputs, dynamics, readout) where {T,I <: AbstractDirectedInterface} =
    ContinuousMachine{T,I}(I(ninputs, noutputs), ContinuousDirectedSystem{T}(nstates, dynamics, readout))

ContinuousMachine{T,N}(ninputs, nstates, noutputs, dynhamics, readout) where {T,N} =
    ContinuousMachine{T, DirectedVectorInterface{T, N}}(ninputs, nstates, noutputs, dynhamics, readout)

ContinuousMachine{T,I}(ninputs::Int, nstates::Int, dynamics) where {T,I} =
    ContinuousMachine{T,I}(ninputs, nstates, nstates, dynamics, (u,p,t) -> u)

InstantaneousContinuousMachine{T}(ninputs, nstates, noutputs, dynamics, readout, dependency) where {T} = 
    ContinuousMachine{T, InstantaneousDirectedInterface{T}}(InstantaneousDirectedInterface{T}(ninputs, noutputs, dependency), ContinuousDirectedSystem{T}(nstates, dynamics, readout))

InstantaneousContinuousMachine{T,I}(ninputs, nstates, noutputs, dynamics, readout, dependency) where {T,I<:InstantaneousDirectedInterface{T}} = 
    ContinuousMachine{T,I}(I(ninputs, noutputs, dependency), ContinuousDirectedSystem{T}(nstates, dynamics, readout))

InstantaneousContinuousMachine{T,N}(ninputs, nstates, noutputs, dynamics, readout, dependency) where {T,N} = 
    ContinuousMachine{T, InstantaneousDirectedVectorInterface{T,N}}(InstantaneousDirectedVectorInterface{T,N}(ninputs, noutputs, dependency), ContinuousDirectedSystem{T}(nstates, dynamics, readout))

InstantaneousContinuousMachine{T}(f::Function, ninputs::Int, noutputs::Int, dependency = nothing) where {T} = 
  ContinuousMachine{T,InstantaneousDirectedInterface{T}}(ninputs, 0, noutputs, (u,x,p,t)->f(x), (u,x,p,t)->T[], dependency)

InstantaneousContinuousMachine{T,I}(f::Function, ninputs::Int, noutputs::Int, dependency = nothing) where {T,I<:AbstractInstantaneousDirectedInterface{T}} = 
  ContinuousMachine{T,I}(ninputs, 0, noutputs, (u,x,p,t)->f(x), (u,x,p,t)->T[], dependency)

InstantaneousContinuousMachine{T,N}(f::Function, ninputs::Int, noutputs::Int, dependency = nothing) where {T,N} = 
  InstantaneousContinuousMachine{T,N}(ninputs, 0, noutputs, (u,x,p,t)->f(x), (u,x,p,t)->T[], dependency)

InstantaneousContinuousMachine{T}(m::ContinuousMachine{T, I}) where {T, I<:DirectedInterface{T}} = 
    ContinuousMachine{T}(InstantaneousDirectedInterface{T}(input_ports(m), output_ports(m), []), 
                         ContinuousDirectedSystem{T}(nstates(m), dynamics(m), (u,x,p,t) -> readout(m, u, p, t))
    )
    


"""    DelayMachine{T}(ninputs, nstates, noutputs, f, r)

A delay open continuous system. The dynamics function `f` defines an ODE ``\\dot u(t) = f(u(t), x(t), h(p,t), p, t)``, where
``u`` is the state, ``x`` captures the exogenous variables, and ``h`` is a history function.

The readout function `r` may depend on the state, history, parameters and time, so it has the signature ``r(u,h,p,t)``.
If it is left out, then ``r=u``.
"""
const DelayMachine{T,I} = Machine{T, I, DelayDirectedSystem{T}}
const InstantaneousDelayMachine{T,I} = Machine{T, I, DelayDirectedSystem{T}}

DelayMachine{T}(interface::I, system::DelayDirectedSystem{T}) where {T, I<:AbstractDirectedInterface} =
    DelayMachine{T, I}(interface, system)

DelayMachine{T}(ninputs, nstates, noutputs, dynamics, readout) where T =
    DelayMachine{T}(DirectedInterface{T}(ninputs, noutputs), DelayDirectedSystem{T}(nstates, dynamics, readout))

DelayMachine{T,I}(ninputs, nstates, noutputs, dynamics, readout) where {T,I<:AbstractDirectedInterface} =
    DelayMachine{T,I}(I(ninputs, noutputs), DelayDirectedSystem{T}(nstates, dynamics, readout))

DelayMachine{T,N}(ninputs, nstates, noutputs, dynhamics, readout) where {T,N} =
    DelayMachine{T, DirectedVectorInterface{T, N}}(ninputs, nstates, noutputs, dynhamics, readout)

DelayMachine{T,I}(ninputs::Int, nstates::Int, dynamics) where {T,I}  =
    DelayMachine{T,I}(ninputs, nstates, nstates, dynamics, (u,h,p,t) -> u)

InstantaneousDelayMachine{T,I}(ninputs, nstates, noutputs, dynamics, readout, dependency) where {T,I<:InstantaneousDirectedInterface{T}} = 
    DelayMachine{T,I}(I(ninputs, noutputs, dependency), DelayDirectedSystem{T}(nstates, dynamics, readout))

InstantaneousDelayMachine{T,N}(ninputs, nstates, noutputs, dynamics, readout, dependency) where {T,N} = 
    DelayMachine{T, InstantaneousDirectedVectorInterface{T,N}}(InstantaneousDirectedVectorInterface{T,N}(ninputs, noutputs, dependency), DelayDirectedSystem{T}(nstates, dynamics, readout))

InstantaneousDelayMachine{T}(f::Function, ninputs::Int, noutputs::Int, dependency = nothing) where {T} = 
    DelayMachine{T,InstantaneousDirectedInterface{T}}(ninputs, 0, noutputs, (u,x,h,p,t)->f(x), (u,x,h,p,t)->T[], dependency)

InstantaneousDelayMachine{T,I}(f::Function, ninputs::Int, noutputs::Int, dependency = nothing) where {T,I<:AbstractInstantaneousDirectedInterface{T}} = 
    DelayMachine{T,I}(ninputs, 0, noutputs, (u,x,h,p,t)->f(x), (u,x,h,p,t)->T[], dependency)

InstantaneousDelayMachine{T,N}(f::Function, ninputs::Int, noutputs::Int, dependency = nothing) where {T,N} = 
    InstantaneousDelayMachine{T,N}(ninputs, 0, noutputs, (u,x,h,p,t)->f(x), (u,x,h,p,t)->T[], dependency)

InstantaneousDelayMachine{T}(m::DelayMachine{T, I}) where {T, I<:DirectedInterface{T}} = 
    DelayMachine{T}(InstantaneousDirectedInterface{T}(input_ports(m), output_ports(m), []), 
                         DelayDirectedSystem{T}(nstates(m), dynamics(m), (u,x,h,p,t) -> readout(m, u, p, t))
    )
  

"""    DiscreteMachine{T}(ninputs, nstates, noutputs, f, r)

A directed open discrete dynamical system. The dynamics function `f` defines a discrete update rule ``u_{n+1} = f(u_n, x_n, p, t)``, where
``u_n`` is the state and ``x_n`` is the value of the exogenous variables at the ``n``th time step.

The readout function `r` may depend on the state, parameters and time step, so it must be of the form ``r(u_n,p,n)``.
If it is left out, then ``r_n=u_n``.
"""
const DiscreteMachine{T,I} = Machine{T, I, DiscreteDirectedSystem{T}}
const InstantaneousDiscreteMachine{T} = Machine{T, InstantaneousDirectedInterface{T}, DiscreteDirectedSystem{T}}

DiscreteMachine{T}(interface::I, system::DiscreteDirectedSystem{T}) where {T, I<:AbstractDirectedInterface} = 
    DiscreteMachine{T, I}(interface, system)

DiscreteMachine{T}(ninputs, nstates, noutputs, dynamics, readout) where T = 
    DiscreteMachine{T}(DirectedInterface{T}(ninputs, noutputs), DiscreteDirectedSystem{T}(nstates, dynamics, readout))

DiscreteMachine{T,I}(ninputs, nstates, noutputs, dynamics, readout) where {T,I<:AbstractDirectedInterface} = 
    DiscreteMachine{T,I}(I(ninputs, noutputs), DiscreteDirectedSystem{T}(nstates, dynamics, readout))

DiscreteMachine{T, N}(ninputs, nstates, noutputs, dynhamics, readout) where {T,N} = 
    DiscreteMachine{T, DirectedVectorInterface{T, N}}(ninputs, nstates, noutputs, dynhamics, readout)

DiscreteMachine{T,I}(ninputs::Int, nstates::Int, dynamics) where {T,I}  = 
    DiscreteMachine{T,I}(ninputs, nstates, nstates, dynamics, (u,p,t) -> u)

DiscreteMachine{T,I}(ninputs, nstates, noutputs, dynamics, readout, dependency) where {T, I<:InstantaneousDirectedInterface{T}} = 
    DiscreteMachine{T, I}(I(ninputs, noutputs, dependency), DiscreteDirectedSystem{T}(nstates, dynamics, readout))

InstantaneousDiscreteMachine{T}(ninputs, nstates, noutputs, dynamics, readout, dependency) where T = 
    DiscreteMachine{T}(InstantaneousDirectedInterface{T}(ninputs, noutputs, dependency), DiscreteDirectedSystem{T}(nstates, dynamics, readout))
  
InstantaneousDiscreteMachine{T}(f::Function, ninputs::Int, noutputs::Int, dependency = nothing) where T = 
    InstantaneousDiscreteMachine{T}(ninputs, 0, noutputs, (u,x,p,t)->T[], (u,x,p,t)->f(x), dependency)

InstantaneousDiscreteMachine(m::DiscreteMachine{T, I}) where {T, I<:DirectedInterface{T}} = 
    DiscreteMachine{T}(InstantaneousDirectedInterface{T}(input_ports(m), output_ports(m), []), 
                         DiscreteDirectedSystem{T}(nstates(m), dynamics(m), (u,x,p,t) -> readout(m, u, p, t))
    )

show(io::IO, vf::ContinuousMachine) = print(io,
    "ContinuousMachine(ℝ^$(nstates(vf)) × ℝ^$(ninputs(vf)) → ℝ^$(nstates(vf)))")
show(io::IO, vf::DelayMachine) = print(io,
    "DelayMachine(ℝ^$(nstates(vf)) × ℝ^$(ninputs(vf)) → ℝ^$(nstates(vf)))")
show(io::IO, vf::DiscreteMachine) = print(io,
    "DiscreteMachine(ℝ^$(nstates(vf)) × ℝ^$(ninputs(vf)) → ℝ^$(nstates(vf)))")

eltype(::AbstractMachine{T}) where T = T


readout(f::AbstractMachine, u::AbstractVector, p = nothing, t = 0) = readout(f)(u, p, t)
readout(f::AbstractMachine, u::FinDomFunction, args...) = readout(f, collect(u), args...)

readout(m::Machine{T,I,S}, u::AbstractVector, x::AbstractVector, p=nothing, t=0) where {T, I<:AbstractInstantaneousDirectedInterface{T}, S} = 
  readout(m)(u,x,p,t)

readout(f::DelayMachine{T,I}, u::AbstractVector, h=nothing, p = nothing, t = 0) where {T, I<:Union{DirectedInterface, DirectedVectorInterface}}= readout(f)(u, h, p, t)
readout(m::DelayMachine{T,I}, u::AbstractVector, x::AbstractVector, h=nothing, p=nothing, t=0) where {T, I<:AbstractInstantaneousDirectedInterface} = 
  readout(m)(u,x,h,p,t)
  

"""    eval_dynamics(m::AbstractMachine, u::AbstractVector, x:AbstractVector{F}, p, t) where {F<:Function}
    eval_dynamics(m::AbstractMachine{T}, u::AbstractVector, x:AbstractVector{T}, p, t) where T

Evaluates the dynamics of the machine `m` at state `u`, parameters `p` and time `t`.
The exogenous variables are set by `x`, which may be a collection either of functions ``x_i(t)`` or of constant values.
In either case, the length of `x` must equal the number of inputs to `m`.
"""
eval_dynamics(f::AbstractMachine, u::AbstractVector, xs::AbstractVector{F}, p=nothing, t=0) where F <: Function =
    eval_dynamics(f, u, [x(t) for x in xs], p, t)

# eval_dynamics(f::AbstractMachine, u::S, xs::T, args...) where {S,T <: Union{FinDomFunction, AbstractVector}} =
#     eval_dynamics(f, collect(u), collect(xs), args...)

eval_dynamics(f::DelayMachine, u, xs, h, p=nothing, t=0) = begin
    ninputs(f) == length(xs) || error("$xs must have length $(ninputs(f)) to set the exogenous variables.")
    dynamics(f)(collect(u), collect(xs), h, p, t)
end

eval_dynamics(f::AbstractMachine, u, xs, p=nothing, t=0) = begin
    ninputs(f) == length(xs) || error("$xs must have length $(ninputs(f)) to set the exogenous variables.")
    dynamics(f)(collect(u), collect(xs), p, t)
end

"""    euler_approx(m::ContinuousMachine, h::Float)

Transforms a continuous machine `m` into a discrete machine via Euler's method with step size `h`. If the dynamics of `m` is given by ``\\dot{u}(t) = f(u(t),x(t),p,t)``, then the dynamics of the new discrete system is given by the update rule ``u_{n+1} = u_n + h f(u_n, x_n, p, t)``.
"""
euler_approx(f::ContinuousMachine{T}, h::Float64) where T = DiscreteMachine{T}(
    ninputs(f), nstates(f), noutputs(f), 
    (u, x, p, t) -> u + h*eval_dynamics(f, u, x, p, t),
    readout(f)
)

"""    euler_approx(m::ContinuousMachine)

Transforms a continuous machine `m` into a discrete machine via Euler's method.
The step size parameter is appended to the end of the system's parameter list.
"""
euler_approx(f::ContinuousMachine{T}) where T = DiscreteMachine{T}(
    ninputs(f), nstates(f), noutputs(f), 
    (u, x, p, t) -> u + p[end]*eval_dynamics(f, u, x, p[1:end-1], t),
    readout(f)
)

"""    euler_approx(ms::Vector{M}, args...) where {M<:ContinuousMachine}
    euler_approx(ms::AbstractDict{S, M}, args...) where {S,M<:ContinuousMachine}

Map `euler_approx` over a collection of machines with identical `args`.
"""
euler_approx(fs::Vector{M}, args...) where {M<:ContinuousMachine} =
    map(f->euler_approx(f,args...), fs)

euler_approx(fs::AbstractDict{S, M}, args...) where {S, M<:ContinuousMachine} =
    Dict(name => euler_approx(f, args...) for (name, f) in fs)






"""    oapply(d::WiringDiagram, ms::Vector{M}) where {M<:AbstractMachine}

Implements the operad algebras for directed composition of dynamical systems, given a
composition pattern (implemented by a directed wiring diagram `d`)
and primitive systems (implemented by a collection of 
machines `ms`). Returns the composite machine.

Each box of the composition pattern `d` must be filled by a machine with the
appropriate type signature.
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

function oapply(d::WiringDiagram, ms::Vector{M}) where {T, I<:InstantaneousDirectedInterface, M<:AbstractMachine{T, I}} 
  isempty(wires(d, input_id(d), output_id(d))) || error("d is not a valid composition syntax because it has pass wires")
  nboxes(d) == length(ms) || error("there are $nboxes(d) boxes but $length(ms) machines")
  for box in 1:nboxes(d)
      fills(ms[box], d, box) || error("$ms[box] does not fill box $box")
  end

  S = coproduct((FinSet∘nstates).(ms))
  dependency_colims = (colimit∘dependency).(ms)
  get_readouts = define_get_readouts(d, dependency_colims)

  return M(input_ports(d), 
    length(apex(S)),
    output_ports(d),
    induced_dynamics(d, ms, S, get_readouts),
    induced_readout(d, ms, S, get_readouts), 
    induced_dependency(d, dependency_colims)
  )

end


"""    oapply(d::WiringDiagram, m::AbstractMachine)

A version of `oapply` where each box of `d` is filled with the same machine `m`.
"""
function oapply(d::WiringDiagram, x::AbstractMachine)
    oapply(d, collect(repeated(x, nboxes(d))))
end

"""    oapply(d::WiringDiagram, generators::AbstractDict{S,M}) where {S,M<:AbstractMachine}

A version of `oapply` where `generators` is a dictionary mapping the name of each box to its corresponding machine. 
"""
function oapply(d::WiringDiagram, ms::AbstractDict{S,M}) where {S, M <: AbstractMachine}
    oapply(d, [ms[box.value] for box in boxes(d)])
end



### Helper functions for `oapply`

function induced_dynamics(d::WiringDiagram, ms::Vector{M}, S, get_readouts = get_readouts) where {T,I, M<:AbstractMachine{T,I}}

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

function induced_dynamics(d::WiringDiagram, ms::Vector{M}, S, get_readouts = get_readouts) where {T,I, M<:DelayMachine{T,I}}

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

function induced_dynamics(d::WiringDiagram, ms::Vector{M}, S, get_readouts = get_readouts) where {T,I<:InstantaneousDirectedInterface{T}, M<:AbstractMachine{T,I}}

  function v(u::AbstractVector, xs::AbstractVector, p, t::Real)  
      states = destruct(S, u) # a list of the states by box
      readouts = get_readouts(ms, states, xs, p, t)
    
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
    
function induced_readout(d::WiringDiagram, ms::Vector{M}, S, get_readouts = get_readouts) where {T, I, M<:AbstractMachine{T,I}}
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

function induced_readout(d::WiringDiagram, ms::Vector{M}, S, get_readouts = get_readouts) where {T, I<:InstantaneousDirectedInterface{T}, M<:AbstractMachine{T,I}}
  function r(u::AbstractVector, xs, p, t)
    states = destruct(S, u)
    readouts = get_readouts(ms, states, xs, p, t)

    map(1:length(output_ports(d))) do p
      sum(map(in_wires(d, output_id(d), p)) do w
        readouts[w.source.box][w.source.port]
      end; init = zero(I))
    end
  end 
end

function induced_readout(d::WiringDiagram, ms::Vector{M}, S, get_readouts = get_readouts) where {T, I, M<:DelayMachine{T,I}}
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

Checks if `m` is of the correct signature to fill box `b` of the wiring diagram `d`.
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

# A function which iteratively produces the readouts for an composite of InstantaneousDirected interface
function define_get_readouts(d::WiringDiagram, dependency_colims::AbstractVector{C}) where {C<:AbstractColimit}

  _, vertex_box, sorted_vs, _ = dependency_graph(d, dependency_colims)

  function get_readouts(ms::AbstractArray{M}, states, xs, p, t) where {T,I<:InstantaneousDirectedInterface{T},M<:Machine{T, I}} 
    readouts = [zeros(T, noutputs(m)) for m in ms]
    for v in sorted_vs
      b = vertex_box[v]
      inputs = map(1:ninputs(ms[b])) do port 
        sum(map(in_wires(d, b, port)) do w
          ys = w.source.box == input_id(d) ? xs : readouts[w.source.box]
          ys[w.source.port]
        end; init = zero(I))
      end
      readouts[b] = readout(ms[b], states[b], inputs, p, t)
    end
    return readouts
  end
  
end

function dependency_graph(d, dependency_colims) 
  g = Graph()
  vs = map(dependency_colims) do c
    add_vertices!(g, length(apex(c)))
  end
  vertex_box = zeros(Int, nv(g))
  for b in 1:length(vs)
    vertex_box[vs[b]] .= b 
  end 
  map(wires(d, :Wire)) do w
    i = last(legs(dependency_colims[w.source.box]))(w.source.port)  # the output port in the pushout that corresponds to the source port of the wire
    j = first(legs(dependency_colims[w.target.box]))(w.target.port) # the input port in the pushout that corresponds to the target port of the wire
    add_edge!(g, vs[w.source.box][i], vs[w.target.box][j])
  end

  sorted_vs = topological_sort(g) # you can get rid of the vs that are "just" input ports (i.e. on which not ouptut port is dependent)
  return g, vertex_box, sorted_vs, vs
end

function induced_dependency(d, dependency_colims)
  g, _, sorted_vs, vs = dependency_graph(d, dependency_colims)

  # flatten 
  h = Graph(nv(g))
  add_edges!(h, src(g), tgt(g))

  for v in sorted_vs
    for w in inneighbors(h, v)
      for z in inneighbors(h, w)
        add_edge!(h, z, v)
      end
    end
  end

  add_edges!(h, 1:nv(h), 1:nv(h))

  # really awkward
  pins = map(wires(d, :InWire)) do w
    i = first(legs(dependency_colims[w.target.box]))(w.target.port)
    vs[w.target.box][i]
  end
  qins = map(wires(d, :InWire)) do w 
    w.source.port 
  end
  pouts = map(wires(d, :OutWire)) do w
    i = last(legs(dependency_colims[w.source.box]))(w.source.port)  # the output port in the pushout that corresponds to the source port of the wire
    vs[w.source.box][i]
  end
  qouts = map(wires(d, :OutWire)) do w 
    w.target.port 
  end

  pb1 = pullback(FinFunction(Array{Int}(pins), nv(h)), FinFunction(Array{Int}(src(h)), nv(h)))
  pb2 = pullback(FinFunction(Array{Int}(tgt(h)), nv(h)), FinFunction(Array{Int}(pouts), nv(h)))

  pb = pullback(legs(pb1)[2], legs(pb2)[1])

  p1 = FinFunction(Array{Int}(qins), length(input_ports(d))) ∘ legs(pb1)[1] ∘ legs(pb)[1]
  p2 = FinFunction(Array{Int}(qouts), length(output_ports(d))) ∘ (legs(pb2)[2] ∘ legs(pb)[2])

  map(1:length(apex(pb))) do i 
    p2(i) => p1(i)
  end |> unique
end




end #module
