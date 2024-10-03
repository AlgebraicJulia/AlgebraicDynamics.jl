module UWDDynam
using Catlab

using Catlab.WiringDiagrams.UndirectedWiringDiagrams: AbstractUWD
import Catlab.WiringDiagrams: oapply, ports

export AbstractResourceSharer, ContinuousResourceSharer, DelayResourceSharer, DiscreteResourceSharer,
euler_approx, nstates, nports, ports, portmap, portfunction, trajectory,
eval_dynamics, eval_dynamics!, exposed_states, fills, induced_states

using Base.Iterators
import Base: show, eltype

### Interface
abstract type AbstractInterface{T} end
abstract type AbstractUndirectedInterface{T} <: AbstractInterface{T} end

struct UndirectedInterface{T} <: AbstractUndirectedInterface{T}
  ports::Vector
end
UndirectedInterface{T}(nports::Int) where T = UndirectedInterface{T}(1:nports)


struct UndirectedVectorInterface{T,N} <: AbstractUndirectedInterface{T}
  ports::Vector
end
UndirectedVectorInterface{T,N}(nports::Int) where {T,N} = UndirectedVectorInterface{T,N}(1:nports)

ports(interface::AbstractUndirectedInterface) = interface.ports
nports(interface::AbstractUndirectedInterface) = length(ports(interface))
ndims(::UndirectedVectorInterface{T, N}) where {T,N} = N

### System dynamics 

abstract type AbstractUndirectedSystem{T} end

struct ContinuousUndirectedSystem{T} <: AbstractUndirectedSystem{T}
  nstates::Int
  dynamics::Function 
  portmap
end

struct DiscreteUndirectedSystem{T} <: AbstractUndirectedSystem{T}
  nstates::Int
  dynamics::Function 
  portmap
end

struct DelayUndirectedSystem{T} <: AbstractUndirectedSystem{T}
  nstates::Int
  dynamics::Function 
  portmap
end

nstates(system::AbstractUndirectedSystem) = system.nstates 
dynamics(system::AbstractUndirectedSystem) = system.dynamics 
portmap(system::AbstractUndirectedSystem) = system.portmap 
portfunction(system::AbstractUndirectedSystem) = FinFunction(portmap(system), nstates(system))
exposed_states(system::AbstractUndirectedSystem, u::AbstractVector) = getindex(u, portmap(system))

"""    ResourceSharer{T}
    ResourceSharer{T,N}

An undirected open dynamical system operating on information of type `T`.
For type arguments `{T,N}`, the system operates on arrays of type `T` and `ndims = N`.
A resource sharer `r` has type signature `r.nports`.
"""
abstract type AbstractResourceSharer{T} end

struct ResourceSharer{T, I, S} <: AbstractResourceSharer{T} 
  interface::I 
  system::S
end

system(r::ResourceSharer) = r.system 
interface(r::ResourceSharer) = r.interface

ports(r::ResourceSharer) = ports(interface(r))
nports(r::ResourceSharer) = nports(interface(r))
nstates(r::ResourceSharer) = nstates(system(r)) 
dynamics(r::ResourceSharer) = dynamics(system(r))
portmap(r::ResourceSharer) = portmap(system(r)) 
portfunction(r::ResourceSharer) = portfunction(system(r))
exposed_states(r::ResourceSharer, u::AbstractVector) = exposed_states(system(r), u)


"""    ContinuousResourceSharer{T}(nports, nstates, f, portmap)

An undirected open continuous system. The dynamics function `f` defines an ODE ``\\dot u(t) = f(u(t),p,t)``, 
where ``u(t)`` has length `nstates`. `nports` is the number of exposed ports and `portmap` is a list of exposed states. 
For example, if `portmap = [2,2,3]` then there are three ports which expose the state variables 2, 2, and 3 respectively.
"""
const ContinuousResourceSharer{T, I} = ResourceSharer{T, I, ContinuousUndirectedSystem}

ContinuousResourceSharer{T, I}(nports, nstates, dynamics, portmap) where {T, I <: AbstractUndirectedInterface} =
  ContinuousResourceSharer{T, I}(UndirectedInterface{T}(nports), ContinuousUndirectedSystem{T}(nstates, dynamics, portmap))

ContinuousResourceSharer{T, N}(nports, nstates, dynamics, portmap) where {T, N} =
  ContinuousResourceSharer{T, UndirectedVectorInterface{T}}(nports, nstates, dynamics, portmap)

ContinuousResourceSharer{T}(interface::I, system::ContinuousUndirectedSystem{T}) where {T, I <: AbstractUndirectedInterface} =
  ContinuousResourceSharer{T, I}(interface, system)

ContinuousResourceSharer{T}(nports, nstates, dynamics, portmap) where {T} =
  ContinuousResourceSharer{T}(UndirectedInterface{T}(nports), ContinuousUndirectedSystem{T}(nstates, dynamics, portmap))
            
"""    ContinuousResourceSharer{T}(nstates, f)
If `nports` and `portmap` are not specified by the user, then it is assumed that `nports` is equal to `nstates` and 
`portmap` is the identity map.
"""
ContinuousResourceSharer{T}(nstates::Int, dynamics::Function) where T =
    ContinuousResourceSharer{T}(nstates,nstates, dynamics, 1:nstates)

"""    DelayResourceSharer{T}(nports, nstates, f, portmap)

An undirected open continuous system. The dynamics function `f` defines a DDE ``\\dot u(t) = f(u(t),h(t),p,t)``,
where ``h`` is a function giving the history of the system's state ``u`` before the interval on which the solution will be computed
begins (usually for ``t < 0``). `f` should have signature ``f(u,h,p,t)``, where ``h`` is a function.
"""
const DelayResourceSharer{T, I} = ResourceSharer{T, I, DelayUndirectedSystem}
DelayResourceSharer{T, I}(nports, nstates, dynamics, portmap) where {T, I <: AbstractUndirectedInterface}= 
  DelayResourceSharer{T, I}(UndirectedInterface{T}(nports), DelayUndirectedSystem{T}(nstates, dynamics, portmap))

DelayResourceSharer{T, N}(nports, nstates, dynamics, portmap) where {T, N} = 
  DelayResourceSharer{T, UndirectedVectorInterface{T}}(nports, nstates, dynamics, portmap)

DelayResourceSharer{T}(interface::I, system::DelayUndirectedSystem{T}) where {T, I <: AbstractUndirectedInterface} =
  DelayResourceSharer{T, I}(interface, system)

DelayResourceSharer{T}(nports, nstates, dynamics, portmap) where {T}= 
  DelayResourceSharer{T}(UndirectedInterface{T}(nports), DelayUndirectedSystem{T}(nstates, dynamics, portmap))

DelayResourceSharer{T}(nstates::Int, dynamics::Function) where T = 
    DelayResourceSharer{T}(nstates,nstates, dynamics, 1:nstates)

"""    DiscreteResourceSharer{T}(nports, nstates, f, portmap)

An undirected open discrete system. The dynamics function `f` defines a discrete update rule ``u_{n+1} = f(u_n, p, t)``.
"""
const DiscreteResourceSharer{T, I} = ResourceSharer{T, I, DiscreteUndirectedSystem}
DiscreteResourceSharer{T, I}(nports, nstates, dynamics, portmap) where {T, I <: AbstractUndirectedInterface}= 
  DiscreteResourceSharer{T, I}(UndirectedInterface{T}(nports), DiscreteUndirectedSystem{T}(nstates, dynamics, portmap))

DiscreteResourceSharer{T, N}(nports, nstates, dynamics, portmap) where {T, N} = 
  DiscreteResourceSharer{T, UndirectedVectorInterface{T}}(nports, nstates, dynamics, portmap)

DiscreteResourceSharer{T}(interface::I, system::DiscreteUndirectedSystem{T}) where {T, I <: AbstractUndirectedInterface} =
  DiscreteResourceSharer{T, I}(interface, system)

DiscreteResourceSharer{T}(nports, nstates, dynamics, portmap) where {T}= 
  DiscreteResourceSharer{T}(UndirectedInterface{T}(nports), DiscreteUndirectedSystem{T}(nstates, dynamics, portmap))

DiscreteResourceSharer{T}(nstates::Int, dynamics::Function) where T = 
    DiscreteResourceSharer{T}(nstates,nstates, dynamics, 1:nstates)

"""    eval_dynamics(r::AbstractResourceSharer, u::AbstractVector, p, t)

Evaluates the dynamics of the resource sharer `r` at state `u`, parameters `p` and time `t`.
Omitting `p` and `t` is allowed if the dynamics of `r` does not depend on them.
"""
eval_dynamics(r::DelayResourceSharer, u::AbstractVector, h, p, t::Real) = dynamics(r)(u, h, p, t)
eval_dynamics!(du, r::DelayResourceSharer, u::AbstractVector, h, p, t::Real) = begin
    du .= eval_dynamics(r, u, h, p, t)
end
eval_dynamics(r::AbstractResourceSharer, u::AbstractVector, p, t::Real) = dynamics(r)(u, p, t)
eval_dynamics!(du, r::AbstractResourceSharer, u::AbstractVector, p, t::Real) = begin
    du .= eval_dynamics(r, u, p, t)
end
eval_dynamics(r::AbstractResourceSharer, u::AbstractVector) = eval_dynamics(r, u, [], 0)
eval_dynamics(r::AbstractResourceSharer, u::AbstractVector, p) = eval_dynamics(r, u, p, 0)

show(io::IO, vf::ContinuousResourceSharer) = print(io,
    "ContinuousResourceSharer(ℝ^$(nstates(vf)) → ℝ^$(nstates(vf))) with $(nports(vf)) exposed port$(nports(vf) > 1 ? "s" : "")")
show(io::IO, vf::DelayResourceSharer) = print(io,
    "DelayResourceSharer(ℝ^$(nstates(vf)) → ℝ^$(nstates(vf))) with $(nports(vf)) exposed port$(nports(vf) > 1 ? "s" : "")")
show(io::IO, vf::DiscreteResourceSharer) = print(io,
    "DiscreteResourceSharer(ℝ^$(nstates(vf)) → ℝ^$(nstates(vf))) with $(nports(vf)) exposed port$(nports(vf) > 1 ? "s" : "")")
eltype(r::AbstractResourceSharer{T}) where T = T

"""    euler_approx(r::ContinuousResourceSharer, h::Float)

Transforms a continuous resource sharer `r` into a discrete resource sharer via Euler's method with step size `h`. If the dynamics of `r` is given by ``\\dot{u}(t) = f(u(t),p,t)``, then the dynamics of the new discrete system is given by the update rule ``u_{n+1} = u_n + h f(u_n, p, t)``.
"""
euler_approx(f::ContinuousResourceSharer{T}, h::Float64) where T = DiscreteResourceSharer{T}(
        nports(f), nstates(f), 
        (u, p, t) -> u + h*eval_dynamics(f, u, p, t),
        portmap(f)
)

"""    euler_approx(r::ContinuousResourceSharer)

Transforms a continuous resource sharer `r` into a discrete resource sharer via Euler's method.
The step size parameter is appended to the end of the system's parameter list.
"""
euler_approx(f::ContinuousResourceSharer{T}) where T = DiscreteResourceSharer{T}(
    nports(f), nstates(f), 
    (u, p, t) -> u + p[end]*eval_dynamics(f, u, p[1:end-1], t),
    portmap(f)
)

"""    euler_approx(rs::Vector{R}, args...) where {T,R<:ContinuousResourceSharer{T}}
    euler_approx(rs::AbstractDict{S, R}, args...) where {S,T,R<:ContinuousResourceSharer{T}}

Map `euler_approx` over a collection of resource sharers with identical `args`.
"""
euler_approx(fs::Vector{R}, args...) where {T, R<:ContinuousResourceSharer{T}} =
    map(f->euler_approx(f,args...), fs)

euler_approx(fs::AbstractDict{S, R}, args...) where {S, T, R<:ContinuousResourceSharer{T}} =
    Dict(name => euler_approx(f, args...) for (name, f) in fs)



# Trajectories
function trajectory() end


"""    fills(r::AbstractResourceSharer, d::AbstractUWD, b::Int)

Checks if `r` is of the correct signature to fill box `b` of the undirected wiring diagram `d`.
"""
function fills(r::AbstractResourceSharer, d::AbstractUWD, b::Int)
    b <= nparts(d, :Box) || error("Trying to fill box $b, when $d has fewer than $b boxes")
    return nports(r) == length(incident(d, b, :box))
end



"""     oapply(d::AbstractUWD, rs::Vector{R}) where {R<:AbstractResourceSharer}

Implements the operad algebras for undirected composition of dynamical systems, given a composition pattern (implemented
by an undirected wiring diagram `d`) and primitive systems (implemented by
a collection of resource sharers `rs`). Returns the composite resource sharer.

Each box of `d` must be filled by a resource sharer of the appropriate type signature. 
"""
oapply(d::AbstractUWD, xs::Vector{R}) where {R <: AbstractResourceSharer} =
    oapply(d, xs, induced_states(d, xs))

"""    oapply(d::AbstractUWD, r::AbstractResourceSharer)

A version of `oapply` where each box of `d` is filled with the same resource sharer `r`.
"""
oapply(d::AbstractUWD, x::AbstractResourceSharer) =
    oapply(d, collect(repeated(x, nboxes(d))))
    
"""     oapply(d::AbstractUWD, generators::AbstractDict{S,R}) where {S,R<:AbstractResourceSharer}

A version of `oapply` where `generators` is a dictionary mapping the name of each box to its corresponding resource sharer.
"""
oapply(d::AbstractUWD, xs::AbstractDict{S,R}) where {S, R <: AbstractResourceSharer} =
    oapply(d, [xs[name] for name in subpart(d, :name)])



function oapply(d::AbstractUWD, xs::Vector{R}, S′::Pushout) where {R <: AbstractResourceSharer}
    
    S = coproduct((FinSet∘nstates).(xs))
    states(b::Int) = legs(S)[b].func

    v = induced_dynamics(d, xs, legs(S′)[1], states)

    junction_map = legs(S′)[2]
    outer_junction_map = FinFunction(subpart(d, :outer_junction), nparts(d, :Junction))

    return R(
        induced_ports(d), 
        length(apex(S′)), 
        v, 
        compose(outer_junction_map, junction_map).func)
end


### Helper functions for `oapply`

induced_ports(d::AbstractUWD) = nparts(d, :OuterPort)
induced_ports(d::RelationDiagram) = subpart(d, [:outer_junction, :variable])

### Returns a pushout where the left leg is the union of all primitive states 
function induced_states(d::AbstractUWD, xs::Vector{R}) where {R <: AbstractResourceSharer}
    for box in parts(d, :Box)
        fills(xs[box], d, box) || error("$(xs[box]) does not fill box $box")
    end
    
    S = coproduct((FinSet∘nstates).(xs))  
    total_portfunction = copair([compose( portfunction(xs[i]), legs(S)[i]) for i in 1:length(xs)])
    
    return pushout(total_portfunction, FinFunction(subpart(d, :junction), nparts(d, :Junction)))
end


function induced_dynamics(d::AbstractUWD, xs::Vector{R}, state_map::FinFunction, states::Function) where {T, R<:ContinuousResourceSharer{T}}
  
    function v(u′::AbstractVector, p, t::Real)
      u = getindex(u′,  state_map.func)
      du = zero(u)
      # apply dynamics
      for b in parts(d, :Box)
        eval_dynamics!(view(du, states(b)), xs[b], view(u, states(b)), p, t)
      end
      # add along junctions
      du′ = [sum(Array{T}(view(du, preimage(state_map, i)))) for i in codom(state_map)]
      return du′
    end

end

function induced_dynamics(d::AbstractUWD, xs::Vector{R}, state_map::FinFunction, states::Function) where {T, R<:DelayResourceSharer{T}}
  
    function v(u′::AbstractVector, h′, p, t::Real)
      u = getindex(u′, state_map.func)
      hist(p,t) = getindex(h′(p,t), state_map.func)
      du = zero(u)
      # apply dynamics
      for b in parts(d, :Box)
        eval_dynamics!(view(du, states(b)), xs[b], view(u, states(b)), (p,t) -> view(hist(p,t), states(b)), p, t)
      end
      # add along junctions
      du′ = [sum(Array{T}(view(du, preimage(state_map, i)))) for i in codom(state_map)]
      return du′
    end

end

function induced_dynamics(d::AbstractUWD, xs::Vector{R}, state_map::FinFunction, states::Function) where {T, R<:DiscreteResourceSharer{T}}
    function v(u′::AbstractVector, p, t::Real)
        u0 = getindex(u′,  state_map.func)
        u1 = zero(u0)
        # apply dynamics
        for b in parts(d, :Box)
          eval_dynamics!(view(u1, states(b)), xs[b], view(u0, states(b)), p, t)
        end
        Δu = u1 - u0
        # add along junctions
        return u′+ [sum(Array{T}(view(Δu, preimage(state_map, i)))) for i in codom(state_map)]
    end
end

end #module
