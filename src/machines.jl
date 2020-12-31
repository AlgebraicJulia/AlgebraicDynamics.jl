module Machines

using Catlab.WiringDiagrams.DirectedWiringDiagrams
using  Catlab.CategoricalAlgebra
using  Catlab.CategoricalAlgebra.FinSets
import Catlab.CategoricalAlgebra: coproduct
import Catlab.WiringDiagrams: oapply

export AbstractMachine, ContinuousMachine, DiscreteMachine, 
nstates, nparams, noutputs, eval_dynamics, readout, euler_approx

using Base.Iterators
import Base: show, eltype

abstract type AbstractMachine{T} end

struct ContinuousMachine{T} <: AbstractMachine{T}
    nparams::Int
    nstates::Int
    noutputs::Int
    dynamics::Function
    readout::Function
end

struct DiscreteMachine{T} <: AbstractMachine{T}
    nparams::Int
    nstates::Int
    noutputs::Int
    dynamics::Function
    readout::Function
end
  
show(io::IO, vf::ContinuousMachine) = print("ContinuousMachine(ℝ^$(vf.nstates) × ℝ^$(vf.nparams) → ℝ^$(vf.nstates))")
show(io::IO, vf::DiscreteMachine) = print("DiscreteMachine(ℝ^$(vf.nstates) × ℝ^$(vf.nparams) → ℝ^$(vf.nstates))")
eltype(m::AbstractMachine{T}) where T = T

nstates(f::AbstractMachine) = f.nstates
nparams(f::AbstractMachine) = f.nparams
noutputs(f::AbstractMachine) = f.noutputs
eval_dynamics(f::AbstractMachine, u, p, args...) = f.dynamics(u,p, args...)
readout(f::AbstractMachine, u, args...) = f.readout(u, args...)

#eulers
euler_approx(f::ContinuousMachine{T}, h::Float64) where T = DiscreteMachine{T}(
    nparams(f), nstates(f), noutputs(f), 
    (u, p, args...) -> u + h*eval_dynamics(f, u, p, args...),
    f.readout
)

euler_approx(f::ContinuousMachine{T}) where T = DiscreteMachine{T}(
    nparams(f), nstates(f), noutputs(f), 
    (u, p, h, args...) -> u + h*eval_dynamics(f, u, p, args...),
    (u, h, args...) -> f.readout(u, args...)
)
euler_approx(fs::Vector{ContinuousMachine{T}}, args...) where T = 
    map(f->euler_approx(f,args...), fs)

# oapply
function fills(m::AbstractMachine, d::WiringDiagram, b::Int)
    b <= nboxes(d) || error("Trying to fill box $b, when $d has fewer than $b boxes")
    b = box_ids(d)[b]
    return nparams(m) == length(input_ports(d,b)) && noutputs(m) == length(output_ports(d,b))
end


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

@inline fillstates!(y, d, xs, States, statefun, paramfun, args...) = colimitmap!(y, States, xs) do i, x
    return x.dynamics(statefun(i), paramfun(i), args...)
end

@inline fillwire(w, d, readouts, Outputs) = readouts[legs(Outputs)[w.source.box - 2](w.source.port)] # FIX - box re-indexing

fillreadins!(readins, d, readouts, Outputs, Params, p) = begin
    for (i,w) in enumerate(wires(d))
        if w.target.box == output_id(d)
            continue
        elseif w.source.box == input_id(d)
            readins[legs(Params)[w.target.box - 2](w.target.port)] += p[w.source.port]
        else
            readins[legs(Params)[w.target.box - 2](w.target.port)] += fillwire(w, d, readouts, Outputs)
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
    Params = coproduct((FinSet∘nparams).(xs))
    Outputs = coproduct((FinSet∘noutputs).(xs))
    ys = zeros(T, length(apex(S)))

    states(u::Vector, b::Int) = u[legs(S)[b](1:xs[b].nstates)]

    v = (u::AbstractVector, p::AbstractVector, args...) -> begin
        readouts = zeros(T, length(apex(Outputs)))
        readins = zeros(T, length(apex(Params)))

        get_states(b) = states(u,b)
        get_params(b) = view(readins, legs(Params)[b](:))
        
        fillreadouts!(readouts, d, xs, Outputs, get_states, args...)
        fillreadins!(readins, d, readouts, Outputs, Params, p)
        fillstates!(ys, d, xs, S, get_states, get_params, args...)
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