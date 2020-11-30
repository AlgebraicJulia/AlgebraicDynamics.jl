using AlgebraicDynamics
using AlgebraicDynamics.CPortGraphs
import AlgebraicDynamics.CPortGraphs: draw
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
using Catlab.Theories

using Test
using PrettyTables

nstates = AlgebraicDynamics.CPortGraphs.nstates

@testset "CPortGraphs" begin

    function printsim(traj, stepfun, indxfun, shape)
        for u in stepfun(traj)
            pretty_table(reshape(indxfun(u), shape), equal_columns_width=true, noheader=true)
        end
    end

    d = barbell(2)
    xs = [
        VectorField{Float64}((x,p, t)->[p[1]*x[1], p[2]*x[2]], x->x, 2, 2),
        VectorField{Float64}((x,p, t)->[1/p[1]*x[1], -p[2]*x[2]], x->x, 2, 2),
    ]
    h = 0.1
    u₀ = ones(Float64, 4)
    composite = oapply(d, xs)
    f, r = composite.update, composite.readout
    @show r(u₀)
    @show u₁ = f(u₀, Float64[], h)
    @show r(u₁)
    f(u₁, [], h)

    @show simulate(composite, 10, h, u₀)
    d₀ = OpenCPortGraph()
    add_parts!(d₀, :B, 1)
    d₁ = barbell(2)
    F = ACSetTransformation((B=[2],), d₀, d₁)
    G = ACSetTransformation((B=[1],), d₀, d₁)
    # |1| <-> |3| <-> |5|
    # |2| <-> |4| <-> |6|
    d₂ = apex(pushout(F,G))
    Catlab.Theories.id(OpenCPortGraph, n) = begin
        g = OpenCPortGraph()
        add_parts!(g, :B, 1)
        add_parts!(g, :P, n, box=1)
        add_parts!(g, :OP, n, con=1:n)
        return g
    end
    lob(n) = let 
        b = barbell(n)
        p = add_parts!(b, :P, n, box=1)
        add_parts!(b, :OP, n, con=p)
        b
    end
    d₂′ = ocompose(barbell(2), [id(OpenCPortGraph, 2), lob(2)])
    @show d₂′
    β = 0.4
    μ = 0.4
    α₁ = 0.01
    α₂ = 0.01

    sirfuncb = (u,p,t)->[-β*u[1]*u[2] - α₁*(u[1]-p[1]), # Ṡ
                         β*u[1]*u[2] - μ*u[2] - α₂*(u[2]-p[2]), #İ
                         μ*u[2] # Ṙ
                         ]
    sirfuncm = (u,p,t)->[-β*u[1]*u[2] - α₁*(u[1]-(p[1]+p[3])/2),
                         β*u[1]*u[2] - μ*u[2] - α₂*(u[2]-(p[2]+p[4])/2),
                         μ*u[2]
                         ]

    boundary  = VectorField{Float64}(sirfuncb, u->u[1:2], 2, 3)
    middle    = VectorField{Float64}(sirfuncm, u->u[[1,2,1,2]], 4, 3)
    threecity = oapply(d₂, [boundary,middle,boundary])

    println("Simulating 3 city")
    traj = simulate(threecity, 100, 0.01, [100,1,0,100,0,0,100,0,0.0])
    map(traj) do u
        return (i1=u[2], i2=u[5], i3=u[8])
    end |> pretty_table

    begin
        # g = OpenCPortGraph()
        # add_parts!(g,  9, :B)
        # add_parts!(g, 21, :P, box=[1,1,2,2,2,3,3,4,4,4,5,5,5,5,6,6,6,7,7,8,8,8,9,9])
        # add_parts!(g, 12, :W, src=1:21, [3,7,])

        g = OpenCPortGraph()
    end

    gl = @acset OpenCPortGraph begin
        B = 3
        P = 7
        W = 4
        OP = 3
        box = [1,1,2,2,2,3,3]
        src = [2, 3, 5, 6]
        tgt = [3, 2, 6, 5] 
        con = [1,4,7]
    end

    gm = @acset OpenCPortGraph begin
        B = 3
        P = 10
        W = 4
        OP = 6
        box = [1,1,1,2,2,2,2,3,3,3]
        src = [2, 4, 6, 8]
        tgt = [4, 2, 8, 6] 
        con = [3, 7, 10, 1, 5, 9]
    end

    gr = @acset OpenCPortGraph begin
        B = 3
        P = 7
        W = 4
        OP = 3
        box = [1,1,2,2,2,3,3]
        src = [1, 3, 4, 6]
        tgt = [3, 1, 6, 4] 
        con = [2,5,7]
    end

    symedges(g) = g.tables.W[g.tables.W.src .<= g.tables.W.tgt]
    sympairs(z) = Iterators.filter(x->x[1] <= x[2], z)
    pg2 = ocompose(barbell(3), [gl, gm])
    g2 = migrate!(Graph(), pg2)
    @test g2[ 9:11, :src] == [1,2,3]
    @test g2[ 9:11, :tgt] == [4,5,6]
    @test g2[12:14, :src] == [4,5,6]
    @test g2[12:14, :tgt] == [1,2,3]

    d3 = ocompose(barbell(3), [id(OpenCPortGraph, 3), lob(3)])
    pg3 = ocompose(d3, [gl,gm,gr])
    g3 = migrate!(Graph(), pg3)
    @test g3[19:24, :src] == 1:6
    @test g3[19:24, :tgt] == [4,5,6,1,2,3]
    @test incident(pg3, 5, :box) == 11:14

    import Base.Iterators: repeated

    α₁ = 1
    fm = VectorField{Float64}((u,p,t) -> α₁ * (sum(p) .- u .* length(p)), u->repeated(u[1], 4), 4, 1)
    fl = VectorField{Float64}((u,p,t) -> α₁ * (sum(p) .- u .* length(p)), u->repeated(u[1], 3), 3, 1)
    fr = VectorField{Float64}((u,p,t) -> α₁ * (sum(p) .- u .* length(p)), u->repeated(u[1], 3), 3, 1)
    ft = VectorField{Float64}((u,p,t) -> α₁ * (sum(p) .- u .* length(p)), u->repeated(u[1], 3), 3, 1)
    fb = VectorField{Float64}((u,p,t) -> α₁ * (sum(p) .- u .* length(p)), u->repeated(u[1], 3), 3, 1)
    fc = VectorField{Float64}((u,p,t) -> α₁ * (sum(p) .- u .* length(p)), u->repeated(u[1], 2), 2, 1)

    @test ft.update(ones(1), ones(3), 0.0) == zeros(1)
    f₁ = oapply(gl, [fc, fl, fc])
    f₂ = oapply(gl, [ft, fm, fb])
    f₃ = oapply(gl, [fc, fr, fc])

    F = oapply(pg3, [fc, fl, fc, ft, fm, fb, fc, fr, fc])
    # @test F.update(ones(Float64, 9), [], 1.0) == zeros(Float64, 9)
    u₀ = zeros(Float64, 9)
    u₀[5] = 1.0
    @test F.update(u₀, [], 1.0)[5] < 0
    @test F.update(u₀, [], 1.0)[4] == α₁
    @test F.update(u₀, [], 1.0)[2] == α₁
    @test F.update(u₀, [], 1.0)[3] == 0

    # traj = simulate(F, 10, 0.1, u₀, [])

    # for u in traj
    #     pretty_table(reshape(u, (3,3)), columns_width=7)
    # end

    d4 = @acset OpenCPortGraph begin
        B = 3
        P = 18
        W = 12
        OP = 6
        box = Iterators.flatten([repeated(i, 6) for i in 1:3])
        con = [1,2,3,16,17,18]
        src = [4,5,6,7,8,9,10,11,12,13,14,15]
        tgt = [7,8,9,4,5,6,13,14,15,10,11,12]
    end
pg3 = ocompose(d4, [gm,gm,gm])
draw(pg3)
pg3[:, :con]

@testset "Laplacians" begin
    @test oapply(gm, [ft, fm, fb]).update(ones(3), ones(6), 1.0) == zeros(3)
    @test oapply(gm, [ft, fm, fb]).update([1,2,1], ones(6), 1.0) == [1,-4,1]
    @test oapply(gm, [ft, fm, fb]).update([1,2,0], ones(6), 1.0) == [1,-5, 4]

    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    @test F.nstates == 9
    @test F.nparams == 6
    @test F.update(ones(9), ones(6), 1.0) == zeros(9)
    @test F.update(2*ones(9), 2*ones(6), 1.0) == zeros(9)

    pg4 = ocompose(d4, [pg3, pg3, pg3])
    pg5 = ocompose(d4, [pg4, pg4, pg4])
    draw(pg5)

    @test (nstates(F),nparams(F)) == (9,6)
    F2 = oapply(d4, [F, F, F])
    @test (nstates(F2),nparams(F2)) == (27,6)
    @test F2.update(ones(Float64, 27), ones(Float64, 6), 0.0) == zeros(27)
    F3 = oapply(d4, [F2,F2,F2])
    @test (nstates(F3), nparams(F3)) == (81, 6)
    @test F3.update(ones(Float64, 81), ones(Float64, 6), 0.0) == zeros(81)

    u₀ = zeros(81)
    u₀[2] = 10
    @time traj = simulate(F3, 50, 0.01, u₀, 10*ones(6))
    @time traj = simulate(F3, 50, 0.01, u₀, 10*ones(6))
    # for u in traj
    #     pretty_table(reshape(u[1:27], (3,9)), equal_columns_width=true, noheader=true)
    # end
end

@testset "Advection-Diffusion" begin

    advecdiffuse(α₁, α₂) = begin
        diffop(u,p,t) = α₁ .* (sum(p) .- u .* length(p))
        advop(u,p,t)  = α₂ .* (p[end] .- u)
        ft = VectorField{Float64}((u,p,t) -> diffop(u,p,u) .+ advop(u,p,t), u->repeated(u[1], 3), 3, 1)
        fm = VectorField{Float64}((u,p,t) -> diffop(u,p,u) .+ advop(u,p,t), u->repeated(u[1], 4), 4, 1)
        fb = VectorField{Float64}((u,p,t) -> diffop(u,p,u) .+ advop(u,p,t), u->repeated(u[1], 3), 3, 1)
        return ft, fm, fb
    end

    ft, fm, fb = advecdiffuse(1.0,2.0)
    ft.update([1.0], [1,1,1.0], 0.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    F.update(zeros(9), ones(6), 0.0)
    traj = simulate(F, 48, 0.1, zeros(9), vcat(ones(3), zeros(3)))
    printsim(traj, t->t[end-2:end], identity, (3,3))

    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 16, 0.1, zeros(27), vcat(ones(3), zeros(3)))
    printsim(traj, t->t[end-2:end], identity, (3,9))

    ft, fm, fb = advecdiffuse(1.0,4.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 16, 0.1, zeros(27), vcat(ones(3), zeros(3)))
    printsim(traj, t->t[end-2:end], identity, (3,9))
    i=16
    @test traj[i][end-2:end][1] == traj[i][end-2:end][2] 
    @test traj[i][end-2:end][2] == traj[i][end-2:end][3] 

    traj = simulate(F2, 16, 0.1, zeros(27), vcat([0,1,0], zeros(3)))
    printsim(traj, t->t[end-2:end], identity, (3,9))
    @test traj[i][end-2:end][1] <= traj[i][end-2:end][2] 
    @test traj[i][end-2:end][2] >= traj[i][end-2:end][3] 
end

@testset "Reaction-Diffusion-Advection" begin


    RDA(α₀, α₁, α₂) = begin
        diffop(u,p,t) = α₁ .* (sum(p) .- u .* length(p))
        advop(u,p,t)  = α₂ .* (p[end] .- u)
        ft = VectorField{Float64}((u,p,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p,t), u->repeated(u[1], 3), 3, 1)
        fm = VectorField{Float64}((u,p,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p,t), u->repeated(u[1], 4), 4, 1)
        fb = VectorField{Float64}((u,p,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p,t), u->repeated(u[1], 3), 3, 1)
        return ft, fm, fb
    end

    ft, fm, fb = RDA(0.1, 1.0,2.0)
    ft.update([1.0], [1,1,1.0], 0.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    F.update(zeros(9), ones(6), 0.0)
    traj = simulate(F, 48, 0.1, zeros(9), vcat(ones(3), zeros(3)))
    printsim(traj, t->t[end-2:end], identity, (3,3))

    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 16, 0.1, zeros(27), vcat(ones(3), zeros(3)))
    printsim(traj, t->t[end-2:end], identity, (3,9))

    ft, fm, fb = RDA(0.1, 1.0,4.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 16, 0.1, zeros(27), vcat(ones(3), zeros(3)))
    printsim(traj, t->t[end-2:end], identity, (3,9))
    i=16
    @test traj[i][end-2:end][1] ≈ traj[i][end-2:end][2] atol=1e-3
    @test traj[i][end-2:end][2] ≈ traj[i][end-2:end][3] atol=1e-3

    traj = simulate(F2, 64, 0.1, zeros(27), vcat([0,1,0], zeros(3)))
    printsim(traj, t->t[end-2:end], identity, (3,9))
    @test traj[i][end-2:end][1] <= traj[i][end-2:end][2] 
    @test traj[i][end-2:end][2] >= traj[i][end-2:end][3] 

    ft, fm, fb = RDA(-0.4, 1.0,4.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 128, 0.1, zeros(27), vcat([0,1,0], zeros(3)))
    printsim(traj, t->t[end-2:end], identity, (3,9))
end


@testset "Grids" begin
    function testgridsize(n,m)
        g = grid(n,m)
        @test nparts(g, :B) == n*m
        @test nparts(g, :P) == 6n + 4n*(m-2)
        @test nparts(g, :W) == 2*((n-1)*m + (m-1)*n)
    end
    for (i,j) in Iterators.product(1:6, 2:4)
        testgridsize(i,j)
    end
end

@testset "Taller Pipe" begin
    G = grid(5,5)
    draw(G)

    RDA(α₀, α₁, α₂) = begin
        diffop(u,p,t) = α₁ .* (sum(p) .- u .* length(p))
        advop(u,p,t)  = begin  α₂ .* (p .- u) end 
        ft = VectorField{Float64}((u,p,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p[2],t), u->repeated(u[1], 3), 3, 1)
        fm = VectorField{Float64}((u,p,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p[2],t), u->repeated(u[1], 4), 4, 1)
        fb = VectorField{Float64}((u,p,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p[2],t), u->repeated(u[1], 3), 3, 1)
        return ft, fm, fb
    end

    function executerda(size, parameters)
        function gensim(n::Int, depth::Int)
            ft,fm,fb = RDA(parameters...)
            F = oapply(apex(meshpath(n)), vcat([ft], collect(repeated(fm, n-2)), [fb]))
            l = 2^(depth-1)
            oapply(apex(gridpath(l, n)), collect(repeated(F, l)))
        end

        function runsim(F, n, depth)
            inputs = zeros(n)
            inputs[1] = 1
            inputs[2] = 1
            inputs[end] = 1
            # inputs[floor(Int, n/2)] = 1 
            l = 2^(depth-1)
            traj = simulate(F, 12, 0.1, zeros(l*n), vcat(inputs, zeros(n)))
            printsim(traj, t->t[end:end], u->u, (n,l))
            return traj
        end
        runsim(gensim(size...), size...)
    end

    executerda((5,4), [0.0,0.0,4.0]);
    executerda((5,4), [0.0,0.1,4.0]);
    executerda((5,4), [0.4,1.0,4.0]);
    executerda((5,4), [0.4,1.0,6.0]);
    executerda((6,5), [0.4,1.0,6.0]);
    @time executerda((32,5), [0.4,1.0,6.0]);
    @time executerda((32,5), [0.4,1.0,6.0]);
end

using Base.Iterators
@testset "RDA Benchmark" begin
    RDA(α₀, α₁, α₂) = begin
        diffop(u,p,t) = α₁ .* (sum(p) .- u .* length(p))
        advop(u,p,t)  = begin  α₂ .* (p .- u) end 
        ft = VectorField{Float64}((u,p,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p[2],t), u->repeated(u[1], 3), 3, 1)
        fm = VectorField{Float64}((u,p,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p[2],t), u->repeated(u[1], 4), 4, 1)
        fb = VectorField{Float64}((u,p,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p[2],t), u->repeated(u[1], 3), 3, 1)
        return ft, fm, fb
    end

    function executerda(size, parameters, nsteps=10, stepsize=0.1)
        n, depth = size
        ft,fm,fb = RDA(parameters...)
        F = oapply(apex(meshpath(n)), vcat([ft], collect(repeated(fm, n-2)), [fb]))
        l = 2^(depth-1)
        F₁ = oapply(apex(gridpath(l, n)), collect(repeated(F, l)))
        inputs = ones(n)
        @time traj = simulate(F₁, nsteps, stepsize, zeros(l*n), vcat(inputs, zeros(n)))
        # printsim(traj, t->t[end:end], u->u, (n,l))
    end
    function executerda_ocomposed(size, parameters, nsteps=10, stepsize=0.1)
        n, depth = size
        l = 2^(depth-1)
        ft,fm,fb = RDA(parameters...)
        G = grid(n, l)
        F = oapply(G, take(cycle(flatten(([ft], repeated(fm, n-2), [fb]))), nparts(G, :B))|> collect)
        inputs = ones(n)
        @time traj = simulate(F, nsteps, stepsize, zeros(l*n), vcat(inputs, zeros(n)))
        # printsim(traj, t->t[end:end], u->u, (n,l))
    end
    println("Benchmark Nested")
    executerda((64,7), [0.4,1.0,4.0], 10, 0.1)
    executerda((64,7), [0.4,1.0,4.0], 10, 0.1)
    executerda((64,7), [0.4,1.0,4.0], 10, 0.1)

    println("Benchmark Flattened")
    executerda_ocomposed((64,7), [0.4,1.0,4.0], 10, 0.1)
    executerda_ocomposed((64,7), [0.4,1.0,4.0], 10, 0.1)
    executerda_ocomposed((64,7), [0.4,1.0,4.0], 10, 0.1)
end
end
