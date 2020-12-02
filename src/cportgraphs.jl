module CPortGraphs
using Catlab
using Catlab.WiringDiagrams
using Catlab.WiringDiagrams.CPortGraphs
using  Catlab.CategoricalAlgebra
using  Catlab.CategoricalAlgebra.FinSets
import Catlab.CategoricalAlgebra: coproduct
import Catlab.WiringDiagrams: oapply
using Catlab.Graphs
import Catlab.Graphs: Graph
import Catlab.CategoricalAlgebra.CSets: migrate!
using Catlab.Graphics
using Catlab.Graphics.Graphviz
using Base.Iterators

export VectorField, nstates, nparams, simulate, barbell, meshpath, gridpath, grid

import Base: show


migrate!(g::Graph, pg::OpenCPortGraph) = migrate!(g, migrate!(CPortGraph(), pg))

draw(g::Graph) = to_graphviz(g, prog="neato", edge_labels=true, node_labels=true)
draw(pg::OpenCPortGraph) = draw(migrate!(Graph(), pg))

concat(xs::Vector) = (collect ∘ Iterators.flatten)(xs)


struct VectorField{T}
    update::Function
    readout::Function
    nparams::Int
    nstates::Int
end

show(io::IO, vf::VectorField) = print("VectorField(ℝ^$(vf.nstates) × $(vf.nparams) → ℝ^$(vf.nstates))")

nstates(f::VectorField) = f.nstates
nparams(f::VectorField) = f.nparams

function feeders(d::OpenCPortGraph, b::Int)
    ports = incident(d, b, :box)
    out = Int[]
    for p in ports
        wires = incident(d, p, :tgt)
        feeders = d[wires, :src]
        for f in feeders
            push!(out, f)
        end
    end
    return out
end

nports(d::OpenCPortGraph, b::Int) = incident(d, b, :box) |> length
nports(d::OpenCPortGraph, b) = map(length, incident(d, b, :box))
nports(d::OpenCPortGraph, b::Colon) = map(length, incident(d, :, :box))

#colimitmap!(f::Function, output, C::Colimit, input) = begin
#    for (i,x) in enumerate(input)
#        y = f(i, x)
#        I = legs(C)[i](1:length(y))
#        length(legs(C)[i].func) == length(y) || error("colimitmap! attempting to fill $(length(legs(C)[i].func)) slots with $(length(y)) values at $i: $(legs(C)[i].func)")
#        output[I] .= y
#    end
#    return output
#end

# Made colimitmap! actually work in-place
# TODO: This expects functions it's calling to handle the possible difference
# in size between the legs of the colimit and the output size (when using for
# fillreadouts, the readout function has to make sure to only affect it's
# output ports, not its input port, even though the cospan is for all ports)
colimitmap!(f::Function, output, C::Colimit, input) = begin
    for (i,x) in enumerate(input)
        I = legs(C)[i].func
        f(view(output, I), i, x)
        #I = legs(C)[i](1:length(y))
        #length(legs(C)[i].func) == length(y) || error("colimitmap! attempting to fill $(length(legs(C)[i].func)) slots with $(length(y)) values at $i: $(legs(C)[i].func)")
        #output[I] .= y
    end
    return output
end

# TODO: Probably want an out-of-place option for fillreadouts and fillstates
@inline fillreadouts!(y, d, xs, Ports, statefun) = colimitmap!(y, Ports, xs) do du,i,x
    val = x.readout(du, statefun(i))
end

@inline fillstates!(y, d, xs, States, statefun, paramfun, t) = colimitmap!(y, States, xs) do du, i, x
    return x.update(du, statefun(i), paramfun(i), t)
end

function oapply(d::OpenCPortGraph, x::VectorField)
    oapply(d, collect(repeated(x, nparts(d, :B))))
end

fillreadins!(readins, ind_map, readouts) = begin
    for i in 1:length(ind_map)
      if ind_map[i] != 0
        readins[ind_map[i]] += readouts[i]
      end
    end
    return readins
end
#=
fillreadins!(readins, d, readouts) = begin
    for b in parts(d, :B)
        ports = incident(d, b, :box)
        for p in ports
            ws = incident(d, p, :tgt)
            qs = d[ws, :src]
            readins[p] += sum(readouts[qs])
        end
    end
    return readins
end=#

function oapply(d::OpenCPortGraph, xs::Vector{VectorField{T}}) where T
    x -> FinSet(x.nstates)
    S = coproduct((FinSet∘nstates).(xs))
    Params = coproduct((FinSet∘nparams).(xs))
    Ports = coproduct([FinSet.(nports(d, b)) for b in parts(d, :B)])
    state(u::Vector, b::Int) = view(u, legs(S)[b](1:xs[b].nstates))
    readouts = zeros(T, length(apex(Ports)))
    readins  = zeros(T, length(apex(Ports)))
    ind_map = zeros(Int, length(apex(Ports)))
    for b in parts(d, :B)
        ports = incident(d, b, :box)
        for p in ports
            ws = incident(d, p, :tgt)
            qs = d[ws, :src]
            ind_map[qs] .= p
        end
    end

    port_to_box = [incident(d, b, :box) for b in 1:nparts(d, :B)]
    ϕ = zeros(T, length(apex(S)))

    # TODO: These are not in-place, and so nesting multiple oapplys is temporarily broken.
    # Flat examples do work.
    υ = (u::AbstractVector, p::AbstractVector, t::Real) -> begin
        # length(p) == length(d[:, :con]) || error("Expected $(length(d[:, :con])) parameters, have $(length(p))")
        statefun(b) = state(u,b)
        paramfun(b) = view(readins, port_to_box[b])
        fillreadouts!(readouts, d, xs, Ports, statefun)
        # communicate readouts to the ports at the other end of the wires, external connections directly fill ports
        readins .= 0
        fillreadins!(readins, ind_map, readouts)
        readins[d[:, :con]] .+= p
        fillstates!(ϕ, d, xs, S, statefun, paramfun, t)
        return ϕ
    end
    function readout(u)
        statefun(b) = state(u,b)
        fillreadouts!(readouts, d, xs, Ports, statefun)
        return readouts[d[:, :con]]
    end
    return VectorField{T}(υ,readout, nparts(d, :OP), apex(S).set)
end


barbell(k::Int) = begin
  g = OpenCPortGraph()
  add_parts!(g, :B, 2)
  add_parts!(g, :P, 2k; box=[fill(1,k); fill(2,k)])
  add_parts!(g, :W, k; src=1:k, tgt=k+1:2k)
  add_parts!(g, :W, k; tgt=1:k, src=k+1:2k)
  return g
end

meshpath(n::Int) = begin
    gt = @acset OpenCPortGraph begin
        B = 1
        P = 3
        W = 0
        OP = 2
        box= ones(Int, 3)
        con= [3,2]
    end
    gm = @acset OpenCPortGraph begin
        B = 1
        P = 4
        W = 0
        OP = 2
        box= ones(Int, 4)
        con= [4,2]
    end
    subs = [gt]
    for i in 2:n-1
        push!(subs, gm)
    end
    push!(subs, gt)
    X = coproduct(subs)
    for i in 1:n-1
        xi = subs[i]
        xj = subs[i+1]
        p = legs(X)[i][:P](nparts(xi, :P)-1)
        q = legs(X)[i+1][:P](1)
        add_parts!(apex(X), :W, 2, src=[p,q], tgt=[q,p])
    end
    c₁ = apex(X)[1:2:nparts(apex(X),:OP) ,:con]
    c₂ = apex(X)[2:2:nparts(apex(X),:OP) ,:con]
    apex(X)[:,:con] = vcat(c₁,c₂)
    return X
end

function gridpath(n::Int, width::Int)
    node = @acset OpenCPortGraph begin
        B = 1
        P = 0
        W = 0
        box = 1
    end
    add_parts!(node, :P, 2width, box=1)
    X = coproduct(collect(repeated(node, n)))
    L = legs(X)
    A = apex(X)
    for i in 1:n-1
        for j in 1:width
            s = L[i][:P](j)
            t = L[i+1][:P](j+width)
            add_part!(A, :W, src=s, tgt=t)
            add_part!(A, :W, src=t, tgt=s)
        end
    end
    upstream = L[1][:P](width+1:2width)
    add_parts!(A, :OP, width, con=upstream)
    downstream = L[end][:P](1:width)
    add_parts!(A, :OP, width, con=downstream)
    return X
end

grid(n::Int, m::Int) = ocompose(apex(gridpath(n,m)), collect(repeated(apex(meshpath(m)), n)))

function simulate(f::VectorField{T}, nsteps::Int, h::Real, u₀::Vector, params=T[]) where T
    t = 0
    us = [u₀]
    for i in 2:nsteps
        u₀ = us[i-1]
        u₁ = u₀ .+ h*f.update(u₀, params, t)
        push!(us, u₁)
        t += h
    end
    return us
end
end
