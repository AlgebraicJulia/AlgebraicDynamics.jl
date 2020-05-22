using Catlab
using Catlab.Doctrines
using Catlab.WiringDiagrams
using Catlab.Programs
using Catlab.Graphics
import Convex, SCS

module Systems
using Catlab
using Catlab.Doctrines
using Catlab.WiringDiagrams
using Catlab.Programs
using Catlab.Graphics
import Convex, SCS
using RecursiveArrayTools
using Test

const FreeSMC = FreeSymmetricMonoidalCategory
struct System{T, F}
    inputs::Vector{Int}
    states::Vector{T}
    outputs::Vector{Int}
    f::F
end

dom_ids(f::FreeSMC.Hom{:generator}) = f.args[1].inputs
codom_ids(f::FreeSMC.Hom{:generator}) = f.args[1].outputs

nstates(f::FreeSMC.Hom{:generator}) = length(f.args[1].states)
nstates(f::FreeSMC.Hom{:otimes}) = sum(nstates(sys) for sys in f.args)
nstates(f::FreeSMC.Hom{:compose}) = sum(nstates(sys) for sys in f.args)

dom_ids(f::FreeSMC.Hom{:otimes}) = ArrayPartition(
  dom_ids(f.args[1]),
  dom_ids(f.args[2]) .+ nstates(f.args[1])
)
codom_ids(f::FreeSMC.Hom{:otimes}) = ArrayPartition(
  codom_ids(f.args[1]),
  codom_ids(f.args[2]) .+ nstates(f.args[1])
)
dom_ids(f::FreeSMC.Hom{:compose}) = dom_ids(f.args[1])
codom_ids(f::FreeSMC.Hom{:compose}) = codom_ids(f.args[2]) .+ nstates(f.args[1])

function states(f::FreeSMC.Hom{:generator})
    return f.args[1].states
end

function states(f::FreeSMC.Hom{:id})
    return ArrayPartition(zeros(Float64, ndims(f.args[1])))
end

function states(composite::FreeSMC.Hom{:compose})
    #@assert foldl(==, map(states, composite.args)) "initial conditions are incorrectly specified"
    return ArrayPartition((map(states, composite.args))...)
end

function states(product::FreeSMC.Hom{:otimes})
    return ArrayPartition((map(states, product.args))...)
end

function forward(sys::System, u, t)
    du = sys.f(u, t)
end

function forward(f::FreeSMC.Hom{:generator}, u, t)
    sys = f.args[1]
    return forward(sys, u, t)
end

function forward(f::FreeSMC.Hom{:id}, u, t)
    return zero(u)
end

function forward(composite::FreeSMC.Hom{:compose}, u, t)
    f,g = composite.args[1], composite.args[2]
    duf = forward(f, u.x[1], t)
    dug = forward(g, u.x[2], t)

    dufup = duf[codom_ids(f)]
    dugup = dug[dom_ids(g)]
    duf[codom_ids(f)] .+= dugup
    dug[  dom_ids(g)] .+= dufup
    du = ArrayPartition(duf, dug)
    return du
end
function forward(product::FreeSMC.Hom{:otimes}, u, t)
    f,g = product.args[1], product.args[2]
    duf = forward(f, u.x[1], t)
    dug = forward(g, u.x[2], t)
    du = ArrayPartition(duf, dug)
    return du
end

initforward(f, t) = forward(f, states(f), t)

using OrdinaryDiffEq
problem(f::FreeSMC.Hom{T}, tspan) where {T} = ODEProblem((u,p,t)->forward(f,u,t), states(f), tspan)

using Test
R = Ob(FreeSMC, Float64)
id(R)
sinesys = System(Int[], [1.0], [1], x -> -x)

@testset "ArrayPartition" begin
    v = ArrayPartition([1,2,3],[4,5,6])
    @test v[1] == 1
    @test v[2] == 2
    @test v[5] == 5
    @test v.x[1] == [1,2,3]
    @test v.x[2] == [4,5,6]

    v = ArrayPartition([1,2,3], ArrayPartition([4,5],[6]), [7,8,9])
    @test v[5] == 5
    @test v[6] == 6
end

@testset "dom_ids" begin
    f = Hom(System([1], [1.0], [1], (x, t)->-x), R, R)
    @test dom_ids(compose(f,f)) == [1]
    @test codom_ids(compose(f,f)) == [2]

    @test dom_ids(otimes(f,f)) == [1, 2]
    @test codom_ids(otimes(f,f)) == [1, 2]

    @test dom_ids(otimes(compose(f,f),f)) == [1, 3]
    @test codom_ids(otimes(compose(f,f),f)) == [2, 3]
end

@testset "Initforward" begin
    @test initforward(id(R), 1.0) == zeros(Float64, 1)
    @test initforward(id(otimes(R, R)), 1.0) == zeros(Float64, 2)
    @test initforward(id(otimes(R, otimes(R, R))), 1.0) == zeros(Float64, 3)

    f = Hom(System([1], [1.0], [1], (x, t)->-x), R, R)
    @test initforward(f, 1) == [-1]
    @test initforward(compose(f,f), 1) == [-2, -2]
    @test initforward(otimes(f,f), 1) == [-1,-1]
    @test initforward(otimes(compose(f,f),f), 1) == [-2, -2,-1]
end

@testset "SIR" begin
    si = Hom(System([1], [99, 1], [2], (x,t)->[-0.0035x[1]*x[2], 0.0035x[1]*x[2]]), R,R)
    ir = Hom(System([1], [1, 0], [2], (x,t)->[-0.05x[1], 0.05x[1]]), R, R)
    sir = compose(si, ir)
    sirtest = Hom(System([1], [99,1,0], [4], (x,t)->[-0.0035x[1]*x[2], 0.0035x[1]*x[2]+ -0.05x[2], 0.05x[2]]), R, R)

    @test collect(initforward(sir, 1))[[1,2,4]] == initforward(sirtest, 1)

    @test states(si) == [99, 1]
    @test states(ir) == [1, 0]
    @test states(sir) == [99, 1, 1, 0]
    @test states(sir)[1] == 99
    @test states(sir)[2] == 1
    @test states(sir)[3] == 1

    @test length(forward(sir, states(sir), 0)) == 4
end
@testset "ODESolvers" begin

    si = Hom(System([1], [99.0, 1], [2], (x,t)->[-0.0035x[1]*x[2], 0.0035x[1]*x[2]]), R,R)
    ir = Hom(System([1], [1.00, 0], [2], (x,t)->[-0.05x[1], 0.05x[1]]), R, R)
    sir = compose(si, ir)
    p = problem(sir, (0,270.0))
    sol = solve(p, alg=Tsit5())
    @test sol.u[end][1] < 1e-1
    @test sol.u[end][end] > 100-1

    si = Hom(System([1], [99.0, 1], [2], (x,t)->[-0.0035x[1]*x[2], 0.0035x[1]*x[2]]), R,R)
    ir = Hom(System([1], [1.00, 0], [2], (x,t)->[-0.05x[1], 0.05x[1]]), R, R)
    sir = compose(si, ir)
    si_2 = Hom(System([1], [99.0, 1], [2], (x,t)->[-0.0025x[1]*x[2], 0.0025x[1]*x[2]]), R,R)
    ir_2= Hom(System([1], [1.00, 0], [2], (x,t)->[-0.07x[1], 0.07x[1]]), R, R)
    sir_2 = compose(si_2, ir_2)
    p = problem(otimes(sir, sir_2), (0,270.0))
    sol = solve(p, alg=Tsit5())
    @test sol.u[end][1] < 1e-1
    @test sol.u[end][4] > 100-1
    @test sol.u[end][5] < 5
    @test sol.u[end][end] > 100-5
end
end
