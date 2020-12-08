module Machines

using Catlab.WiringDiagrams.DirectedWiringDiagrams
using  Catlab.CategoricalAlgebra
using  Catlab.CategoricalAlgebra.FinSets
import Catlab.CategoricalAlgebra: coproduct
import Catlab.WiringDiagrams: oapply
export Machine, oapply

struct Machine{T}
    nparams::Int
    nstates::Int
    noutputs::Int
    update::Function
    readout::Function
end
  
show(io::IO, vf::Machine) = print("Machine(ℝ^$(vf.nstates) × ℝ^$(vf.nparams) → ℝ^$(vf.nstates))")

nstates(f::Machine) = f.nstates
nparams(f::Machine) = f.nparams
noutputs(f::Machine) = f.noutputs

function oapply(d::WiringDiagram, x::Machine)
    oapply(d, collect(repeated(x, nparts(d, :B))))
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

@inline fillstates!(y, d, xs, States, statefun, paramfun) = colimitmap!(y, States, xs) do i, x
    return x.update(statefun(i), paramfun(i))
end

@inline fillwire(w, d, readouts, Outputs, p) = begin
    if w.source.box == input_id(d)
        return p[w.source.port]
    end
    return readouts[legs(Outputs)[w.source.box - 2](w.source.port)] # FIX - box re-indexing
end

fillreadins!(readins, d, readouts, Outputs, Params, p) = begin
    for (i,w) in enumerate(wires(d))
        if w.target.box == output_id(d)
            continue
        end
        readins[legs(Params)[w.target.box - 2](w.target.port)] = fillwire(w, d, readouts, Outputs, p)
    end
    return readins
end

function oapply(d::WiringDiagram, xs::Vector{Machine{T}}) where T
    #nboxes(composite) == length(dynamics)  || error("there are $nboxes(composite) boxes but $length(dynamics) machines")

    S = coproduct((FinSet∘nstates).(xs))

    Params = coproduct((FinSet∘nparams).(xs))
    Outputs = coproduct((FinSet∘noutputs).(xs))

    readouts = zeros(T, length(apex(Outputs)))
    readins = zeros(T, length(apex(Params)))

    ys = zeros(T, length(apex(S)))
    states(u::Vector, b::Int) = u[legs(S)[b](1:xs[b].nstates)]


    v = (u::AbstractVector, p::AbstractVector, t::Real=0.0) -> begin
        get_states(b) = states(u,b)
        get_params(b) = view(readins, legs(Params)[b](:))
        fillreadouts!(readouts, d, xs, Outputs, get_states)
        fillreadins!(readins, d, readouts, Outputs, Params, p)
        fillstates!(ys, d, xs, S, get_states, get_params)
        return ys
    end

    function readout(u::AbstractVector, p::AbstractVector, t::Real=0.0)
        get_states(b) = states(u,b)
        fillreadouts!(readouts, d, xs, Outputs, get_states)
        r = zeros(T, length(d.output_ports))
        for w in in_wires(d, output_id(d))
            r[w.target.port] += fillwire(w, d, readouts, Outputs, p)
        end
        return r
    end

    return Machine{T}(length(d.input_ports), length(apex(S)), length(d.output_ports), v, readout)
    
end

end #module