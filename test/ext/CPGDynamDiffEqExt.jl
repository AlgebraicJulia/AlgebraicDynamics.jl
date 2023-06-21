using AlgebraicDynamics
using Catlab
using DelayDiffEq

# CPGDynam Integration
######################

function simulate(f::ContinuousMachine{T}, nsteps::Int, h::Real, u0::Vector, xs=T[]) where T
    trajectory(euler_approx(f, h), u0, xs, nothing, nsteps)
end

function printsim(traj, stepfun, indxfun, shape)
    for u in stepfun(traj)
        pretty_table(reshape(indxfun(u), shape), equal_columns_width=true, noheader=true)
    end
end

@testset "Laplacians" begin
    @test eval_dynamics(oapply(gm, [ft, fm, fb]), ones(3), ones(6), nothing, 1.0) == zeros(3)
    @test eval_dynamics(oapply(gm, [ft, fm, fb]), [1,2,1], ones(6), nothing, 1.0) == [1,-4,1]
    @test eval_dynamics(oapply(gm, [ft, fm, fb]), [1,2,0], ones(6), nothing, 1.0) == [1,-5, 4]

    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    @test nstates(F) == 9
    @test ninputs(F) == 6
    @test eval_dynamics(F, ones(9), ones(6), nothing, 1.0) == zeros(9)
    @test eval_dynamics(F, 2*ones(9), 2*ones(6), nothing, 1.0) == zeros(9)

    pg4 = ocompose(d4, [pg3, pg3, pg3])
    pg5 = ocompose(d4, [pg4, pg4, pg4])
    # draw(pg5)

    @test (nstates(F),ninputs(F)) == (9,6)
    F2 = oapply(d4, [F, F, F])
    @test (nstates(F2),ninputs(F2)) == (27,6)
    @test eval_dynamics(F2, ones(Float64, 27), ones(Float64, 6), nothing, 0.0) == zeros(27)
    F3 = oapply(d4, [F2,F2,F2])
    @test (nstates(F3), ninputs(F3)) == (81, 6)
    @test eval_dynamics(F3, ones(Float64, 81), ones(Float64, 6), nothing, 0.0) == zeros(81)
end

@testset "Advection-Diffusion" begin
    advecdiffuse(α₁, α₂) = begin
        diffop(u,p,t) = α₁ .* (sum(p) .- u .* length(p))
        advop(u,p,t)  = α₂ .* (p[end] .- u)
        ft = ContinuousMachine{Float64}(3, 1, (u,p,q,t) -> diffop(u,p,u) .+ advop(u,p,t), (u,p,t)->collect(repeated(u[1], 3)))
        fm = ContinuousMachine{Float64}(4, 1, (u,p,q,t) -> diffop(u,p,u) .+ advop(u,p,t), (u,p,t)->collect(repeated(u[1], 4)))
        fb = ContinuousMachine{Float64}(3, 1, (u,p,q,t) -> diffop(u,p,u) .+ advop(u,p,t), (u,p,t)->collect(repeated(u[1], 3)))
        return ft, fm, fb
    end

    ft, fm, fb = advecdiffuse(1.0,2.0)
    eval_dynamics(ft, [1.0], [1,1,1.0], 0.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    eval_dynamics(F, zeros(9), ones(6), 0.0)
    traj = simulate(F, 48, 0.1, zeros(9), vcat(ones(3), zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,3))

    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 16, 0.1, zeros(27), vcat(ones(3), zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,9))

    ft, fm, fb = advecdiffuse(1.0,4.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 16, 0.1, zeros(27), vcat(ones(3), zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,9))
    i=16
    @test traj[i][end-2:end][1] == traj[i][end-2:end][2]
    @test traj[i][end-2:end][2] == traj[i][end-2:end][3]

    traj = simulate(F2, 16, 0.1, zeros(27), vcat([0,1,0], zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,9))
    @test traj[i][end-2:end][1] <= traj[i][end-2:end][2]
    @test traj[i][end-2:end][2] >= traj[i][end-2:end][3]
end

@testset "Reaction-Diffusion-Advection" begin
    RDA(α₀, α₁, α₂) = begin
        diffop(u,p,t) = α₁ .* (sum(p) .- u .* length(p))
        advop(u,p,t)  = α₂ .* (p[end] .- u)
        ft = ContinuousMachine{Float64}(3, 1, (u,p,q,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p,t), (u,p,t)->collect(repeated(u[1], 3)))
        fm = ContinuousMachine{Float64}(4, 1, (u,p,q,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p,t), (u,p,t)->collect(repeated(u[1], 4)))
        fb = ContinuousMachine{Float64}(3, 1, (u,p,q,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p,t), (u,p,t)->collect(repeated(u[1], 3)))
        return ft, fm, fb
    end

    ft, fm, fb = RDA(0.1, 1.0,2.0)
    eval_dynamics(ft, [1.0], [1,1,1.0], nothing, 0.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    eval_dynamics(F, zeros(9), ones(6), nothing, 0.0)
    traj = simulate(F, 48, 0.1, zeros(9), vcat(ones(3), zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,3))

    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 16, 0.1, zeros(27), vcat(ones(3), zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,9))

    ft, fm, fb = RDA(0.1, 1.0,4.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 16, 0.1, zeros(27), vcat(ones(3), zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,9))
    i=16
    @test traj[i][end-2:end][1] ≈ traj[i][end-2:end][2] atol=1e-3
    @test traj[i][end-2:end][2] ≈ traj[i][end-2:end][3] atol=1e-3

    traj = simulate(F2, 64, 0.1, zeros(27), vcat([0,1,0], zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,9))
    @test traj[i][end-2:end][1] <= traj[i][end-2:end][2]
    @test traj[i][end-2:end][2] >= traj[i][end-2:end][3]

    ft, fm, fb = RDA(-0.4, 1.0,4.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 128, 0.1, zeros(27), vcat([0,1,0], zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,9))
end


@testset "Ross-Macdonald model" begin
    c = @acset OpenCPortGraph begin
        Box = 2
        Port = 2
        Wire = 2
        OuterPort = 0
        box = [1,2]
        src = [1,2]
        tgt = [2,1]
        con = []
    end
    dzdt_delay = function(u,x,h,p,t)
        Y, Z = u
        Y_delay, Z_delay = h(p, t - p.n)
        X, X_delay = x[1]

        [p.a*p.c*X*(1 - Y - Z) -
            p.a*p.c*X_delay*(1 - Y_delay - Z_delay)*exp(-p.g*p.n) -
            p.g*Y,
        p.a*p.c*X_delay*(1 - Y_delay - Z_delay)*exp(-p.g*p.n) -
            p.g*Z]
    end
    dxdt_delay = function(u,x,h,p,t)
        X, = u
        Z, _ = x[1]
        [p.m*p.a*p.b*Z*(1 - X) - p.r*X]
    end

    mosquito_delay_model = DelayMachine{Float64, 2}(
        1, 2, 1, dzdt_delay, (u,h,p,t) -> [[u[2], h(p,t - p.n)[2]]])
    human_delay_model = DelayMachine{Float64, 2}(
        1, 1, 1, dxdt_delay, (u,h,p,t) -> [[u[1], h(p, t - p.n)[1]]])
    rm_model = oapply(c, [mosquito_delay_model, human_delay_model])

    params = LVector(a = 0.3, b = 0.55, c = 0.15,
        g = 0.1, n = 10, r = 1.0/200, m = 0.5)

    u0_delay = [0.09, .01, 0.3]
    tspan = (0.0, 365.0*5)
    hist(p,t) = u0_delay;

    prob = DDEProblem(rm_model, u0_delay, [], hist, tspan, params)
    alg = MethodOfSteps(Tsit5())
    sol = solve(prob, alg)
    a, b, c, g, n, r, m = params
    R0 = (m*a^2*b*c*exp(-g*n))/(r*g)
    @test isapprox(last(sol)[3], (R0 - 1)/(R0 + (a*c)/g), atol = 1e-3)
end
