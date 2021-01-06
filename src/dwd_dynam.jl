module DWDDynam
""" This module implements operad algebras corresponding to the directed composition of open dynamical systems.
"""

using Catlab.WiringDiagrams.DirectedWiringDiagrams
using  Catlab.CategoricalAlgebra
using  Catlab.CategoricalAlgebra.FinSets
import Catlab.WiringDiagrams: oapply

import ..UWDDynam: nstates, eval_dynamics, euler_approx

export AbstractMachine, ContinuousMachine, DiscreteMachine, 
nstates, ninputs, noutputs, eval_dynamics, readout, euler_approx

using Base.Iterators
import Base: show, eltype

"""A directed open dynamical system

In the operad algebra, `m::AbstractMachine` has type signature 
(`m.ninputs`, `m.outputs`).
"""
abstract type AbstractMachine{T} end


""" A directed open continuous dynamical system.
"""
struct ContinuousMachine{T} <: AbstractMachine{T}
    ninputs::Int
    nstates::Int
    noutputs::Int
    dynamics::Function
    readout::Function
end

""" A directed open discrete dynamical system.
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
eval_dynamics(f::AbstractMachine, u, p, args...) = f.dynamics(u,p, args...)
readout(f::AbstractMachine, u, args...) = f.readout(u, args...)


"""Transforms a continuous machine into a discrete machine 
via Euler's method.
"""
euler_approx(f::ContinuousMachine{T}, h::Float64) where T = DiscreteMachine{T}(
    ninputs(f), nstates(f), noutputs(f), 
    (u, p, args...) -> u + h*eval_dynamics(f, u, p, args...),
    f.readout
)

euler_approx(f::ContinuousMachine{T}) where T = DiscreteMachine{T}(
    ninputs(f), nstates(f), noutputs(f), 
    (u, p, h, args...) -> u + h*eval_dynamics(f, u, p, args...),
    (u, h, args...) -> f.readout(u, args...)
)
euler_approx(fs::Vector{ContinuousMachine{T}}, args...) where T = 
    map(f->euler_approx(f,args...), fs)

euler_approx(fs::AbstractDict{S, ContinuousMachine{T}}, args...) where {S, T} = 
    Dict(name => euler_approx(f, args...) for (name, f) in fs)

"""Checks if a machine is of the correct signature to fill a 
box in a directed wiring diagram.
"""
function fills(m::AbstractMachine, d::WiringDiagram, b::Int)
    b <= nboxes(d) || error("Trying to fill box $b, when $d has fewer than $b boxes")
    b = box_ids(d)[b]
    return ninputs(m) == length(input_ports(d,b)) && noutputs(m) == length(output_ports(d,b))
end

"""Implements the operad algebras CDS and DDS given a 
composition pattern (implemented by a directed wiring diagram)
and primitive systems (implemented by a collection of 
machines).

Each box of the wiring diagram is filled by a machine with the 
appropriate type signature. Returns the composite machine.
"""
function oapply(d::WiringDiagram, x::AbstractMachine)
    oapply(d, collect(repeated(x, nboxes(d))))
end

function oapply(d::WiringDiagram, xs::AbstractDict)
    oapply(d, [xs[box.value] for box in boxes(d)])
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

@inline fillreadouts!(y, d, xs, Outputs, statefun, args...) = colimitmap!(y, Outputs, xs) do i,x
    return x.readout(statefun(i), args...)
end

@inline fillstates!(y, d, xs, States, statefun, inputfun, args...) = colimitmap!(y, States, xs) do i, x
    return x.dynamics(statefun(i), inputfun(i), args...)
end

@inline fillwire(w, d, readouts, Outputs) = readouts[legs(Outputs)[w.source.box - 2](w.source.port)] # FIX - box re-indexing

fillreadins!(readins, d, readouts, Outputs, Inputs, p) = begin
    for (i,w) in enumerate(wires(d))
        if w.target.box == output_id(d)
            continue
        elseif w.source.box == input_id(d)
            readins[legs(Inputs)[w.target.box - 2](w.target.port)] += p[w.source.port]
        else
            readins[legs(Inputs)[w.target.box - 2](w.target.port)] += fillwire(w, d, readouts, Outputs)
        end
        
    end
    return readins
end

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

    states(u::Vector, b::Int) = u[legs(S)[b](1:xs[b].nstates)]

    v = (u::AbstractVector, p::AbstractVector, args...) -> begin
        readouts = zeros(T, length(apex(Outputs)))
        readins = zeros(T, length(apex(Inputs)))

        get_states(b) = states(u,b)
        get_inputs(b) = view(readins, legs(Inputs)[b](:))
        
        fillreadouts!(readouts, d, xs, Outputs, get_states, args...)
        fillreadins!(readins, d, readouts, Outputs, Inputs, p)
        fillstates!(ys, d, xs, S, get_states, get_inputs, args...)
        return ys
    end

    function readout(u::AbstractVector, args...)
        readouts = zeros(T, length(apex(Outputs)))
        get_states(b) = states(u,b)

        fillreadouts!(readouts, d, xs, Outputs, get_states, args...)
        r = zeros(T, length(d.output_ports))
        for w in in_wires(d, output_id(d))
            r[w.target.port] += fillwire(w, d, readouts, Outputs)
        end
        return r
    end

    return Machine(length(d.input_ports), length(apex(S)), length(d.output_ports), v, readout)
    
end

end #module