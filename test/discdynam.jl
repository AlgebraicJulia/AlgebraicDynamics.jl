using AlgebraicDynamics.DiscDynam
using AlgebraicDynamics.DiscDynam: functor
using Catlab
using LinearAlgebra
using Catlab.CategoricalAlgebra
using Test

@testset "DiscDynam" begin
f(x) = [x[1]*x[2], x[1]+x[2]]
g(x) = [x[1]+x[2], x[2]-x[1], x[3]]

d = DynamUWD()
add_parts!(d, :Junction,  3, jvalue=[1,1,1])
add_parts!(d, :Box,       2, dynamics=[f,g])
add_parts!(d, :State,     5, system=[1,1,2,2,2], value=[1,1,1,1,3])
add_parts!(d, :Port,      4, box=[1, 1, 2, 2], junction=[1, 2, 2, 3], state=[1,2,3,4])
add_parts!(d, :OuterPort, 2, outer_junction=[1,3])

@test isconsistent(d)
@test update!(d) == [1,3,3,0,3]
@test update!(d) == [3,4,4,-3,3]
@test isconsistent(d)


x = [1,1,1,1,3]
y = zero(x)
update!(y, d, x)
@test y == [1,3,3,0,3]
@test get_exposed(d, y) == [1, 0]
y′ = zero(x)

update!(y′, d, y)
@test y′ == [3,4,4,-3,3]
@test get_exposed(d, y′) == [3, -3]

f(x) = [x[2], x[1]]
g(x) = [x[1]+x[2], x[2]-x[1]]
h(x) = [2*x[1], 2*x[1]*x[2]]

d = DynamUWD()
add_parts!(d, :Junction,  3, jvalue=[1,0,5])
add_parts!(d, :Box,       3, dynamics=[f,g,h])
add_parts!(d, :State,     6, system=[1,1, 2,2, 3,3], value=[1,1,1,5,3,5])
add_parts!(d, :Port,      5, box=[1, 1, 2, 2, 3], junction=[1, 1, 1, 3,3], state=[1,2,3,4,6])
add_parts!(d, :OuterPort, 0, outer_junction=Int[])

@assert isconsistent(d) "d is not consistent, check the initial condition"
@test update!(d) == [6, 6, 6, 29, 6, 29]
@assert isconsistent(d) "d is not consistent, check the initial condition"

@testset "Preallocated" begin
    d = DynamUWD{Float64, Function}()
    add_parts!(d, :Junction,  3, jvalue=[1,0,5])
    add_parts!(d, :Box,       3, dynamics=[f,g,h])
    add_parts!(d, :State,     6, system=[1,1, 2,2, 3,3], value=[1,1,1,5,3,5])
    add_parts!(d, :Port,      5, box=[1, 1, 2, 2, 3], junction=[1, 1, 1, 3,3], state=[1,2,3,4,6])
    add_parts!(d, :OuterPort, 0, outer_junction=Int[])
    x = subpart(d, :value)
    y = zero(x)
    update!(y, d, x)
    @test y == [6,6,6,29,6,29]
end

@testset "Recursive DiscDynam" begin
    f(x) = [x[1]*x[2], x[1]+x[2]]
    g(x) = [x[1]+x[2], x[2]-x[1], x[3]]

    d = DynamUWD{Float64, Function}()
    add_parts!(d, :Junction,  3, jvalue=[1,1,1])
    add_parts!(d, :Box,       2, dynamics=[f,g])
    add_parts!(d, :State,     5, system=[1,1,2,2,2], value=[1,1,1,1,3])
    add_parts!(d, :Port,      4, box=[1, 1, 2, 2], junction=[1, 2, 2, 3], state=[1,2,3,4])
    add_parts!(d, :OuterPort, 2, outer_junction=[1,3])


    gdef = DynamUWD{Float64, Function}()
    add_parts!(gdef, :Box, 2, dynamics=[x->[x[1]+x[2], x[2]-x[1]], x->[x[1]]])
    add_parts!(gdef, :State, 3, value=[1,1,3], system=[1,1,2])
    add_parts!(gdef, :Junction, 3)
    add_parts!(gdef, :Port, 3, box=[1,1,2], junction=[1,2,3], state=[1,2,3])
    add_parts!(gdef, :OuterPort, 2, outer_junction=[1,2])
    set_subpart!(gdef, :jvalue, [1,1,3])
    g2 = dynamics(gdef)

    d2 = DynamUWD{Float64, Function}()
    add_parts!(d2, :Junction,  3, jvalue=[1,1,1])
    add_parts!(d2, :Box,       2, dynamics=[f,g2])
    add_parts!(d2, :State,     5, system=[1,1,2,2,2], value=[1,1,1,1,3])
    add_parts!(d2, :Port,      4, box=[1, 1, 2, 2], junction=[1, 2, 2, 3], state=[1,2,3,4])
    add_parts!(d2, :OuterPort, 2, outer_junction=[1,3])

    x = copy(subpart(d2, :value))
    x₁ = update!(d)
    x₂ = update!(d)

    y₁ = update!(d2)
    y₂ = update!(d2)
    @test x₁ == y₁
    @test x₂ == y₂


    d3 = DynamUWD{Float64, Function}()
    add_parts!(d3, :Junction,  3, jvalue=[1,1,1])
    add_parts!(d3, :Box,       2, dynamics=[f,dynamics!(gdef)])
    add_parts!(d3, :State,     5, system=[1,1,2,2,2], value=[1,1,1,1,3])
    add_parts!(d3, :Port,      4, box=[1, 1, 2, 2], junction=[1, 2, 2, 3], state=[1,2,3,4])
    add_parts!(d3, :OuterPort, 2, outer_junction=[1,3])
    @test update!(zero(x), d3, x) == y₁
    @test update!(zero(x), d3, y₁) == y₂

    d3 = DynamUWD{Float64, Union{Function, DynamUWD}}()
    add_parts!(d3, :Junction,  3, jvalue=[1,1,1])
    add_parts!(d3, :Box,       2, dynamics=[f,gdef])
    add_parts!(d3, :State,     5, system=[1,1,2,2,2], value=[1,1,1,1,3])
    add_parts!(d3, :Port,      4, box=[1, 1, 2, 2], junction=[1, 2, 2, 3], state=[1,2,3,4])
    add_parts!(d3, :OuterPort, 2, outer_junction=[1,3])
    @test update!(zero(x), d3, x) == y₁
    @test update!(zero(x), d3, y₁) == y₂
end

@testset "Composition of Systems" begin
    f(x) = [x[1]*x[2], x[1]+x[2]]
    g(x) = [x[1]+x[2], x[2]-x[1], x[3]]
    f_rel = @relation (x, y) where (x, y) begin
        f(x, y)
    end
    g_rel = @relation (x, y) where (x, y) begin
        g(x, y)
    end

    to_dynam = functor(Dict(:f=>Dynam(
                            f,
                            2,
                            [1,2],
                            [1,1]),
                          :g=>Dynam(
                            g,
                            3,
                            [1,2],
                            [1,1,3]))
    );
    f_dyn = to_dynam(f_rel)
    g_dyn = to_dynam(g_rel)

    cosp = Cospan(FinFunction([1,2,2,3], 3), FinFunction([1,3], 3))
    fg_dyn = AlgebraicDynamics.DiscDynam.compose(cosp)(f_dyn, g_dyn)

    output = zeros(5)
    input = [1,1,1,1,3]
    @test isconsistent(fg_dyn)
    @test update!(output, fg_dyn, input) == [1,3,3,0,3]
    input = output
    @test update!(output, fg_dyn, input) == [3,4,4,-3,3]
    @test isconsistent(fg_dyn)
end

#=

Desired API for defining dynamical systems:
we need a syntax for expressing the structure of the problem
d_str = @relation (x, z) where (x::Var, z::Var) begin
    f(x, y)
    g(y, z)
end

# apply a functor to fill in the boxes with concrete dynamical systems

d_cont = Functor(:f=>Dynam(
        dynam=f(x) = [x[1]*x[2]...]
        states=2,
        portmap=(ports=>states)
        values=[-3.0, 1.2])
    :g=>Dynam(g(x) = states,
    ...)
)(d_str)
# junction values can be computed by commutativity
# junction⋅jvalue = state⋅value
# such that isconsistent(d_cont) is satisfied


d = EulerFunctor(h)(d_cont)
# just takes box.dynam => h*box.dynam

for i in 1:100
    update!(d)
end
=#

@testset "Relation Functor" begin
    f(x) = [x[1]*x[2], x[1]+x[2]]
    g(x) = [x[1]+x[2], x[2]-x[1], x[3]]
    d_str = @relation (x, z) where (x, y, z) begin
        f(x, y)
        g(y, z)
    end

    d_cont = functor(Dict(:f=>Dynam(
                            f,
                            2,
                            [1,2],
                            [1,1]),
                          :g=>Dynam(
                            g,
                            3,
                            [1,2],
                            [1,1,3]))
    )(d_str);

    @test isconsistent(d_cont)
    @test update!(d_cont) == [1,3,3,0,3]
    @test update!(d_cont) == [3,4,4,-3,3]
    @test isconsistent(d_cont)
end

@testset "LV Model" begin
    α = 1.2
    β = 0.1
    γ = 1.3
    δ = 0.1

    gen = Dict(
        :birth     => u -> u+[ α*u[1]],
        :death     => u -> u+[-γ*u[1]],
        :predation => u -> u+[-β*u[1]*u[2], δ*u[1]*u[2]],
    )

    d = @relation (x,y) where (x, y) begin
        birth(x)
        predation(x,y)
        death(y)
    end

    lv = functor( Dict(:birth => Dynam(gen[:birth], 1, [1], [0]),
                      :death => Dynam(gen[:death], 1, [1], [0]),
                      :predation => Dynam(gen[:predation], 2, [1,2], [0,0])))(d)

    # Test starting at equlibrium values
    u₀ = [13.0, 13.0, 12.0, 12.0]
    set_values!(lv, u₀)
    @test all([norm(u₀ - update!(lv)) < 1e-4 for i in 1:10])

    # Test non-equlibrium values
    u₀ = [17.0, 17.0, 11.0, 11.0]
    set_values!(lv, u₀)
    @test norm(u₀ - update!(lv)) > 1e-1
end


@testset "dynamics with input" begin
    f(x, a) = [x[1] + a, a*x[2]]
    g(x, a) = [x[2], a-x[1]]

    f_dyn = Dynam(f, 2, [1,2], [-1,1])
    g_dyn = Dynam(g, 2, [1,2], [-1,1])

    cspn = Cospan(FinFunction([1,2,1,2], 2), FinFunction([1,2], 2))

    d = AlgebraicDynamics.DiscDynam.compose(cspn)(f_dyn, g_dyn)

    input = subpart(d,:value)
    output = zero(input)
    @test update!(output,d,input, 1) == [2, 2, 2, 2]
    @test update!(output,d,input, 100) == [101, 200, 101, 200]


    h(x,a,b) = [a*x[1], b*x[2]]
    k(x,a,b) = [a*b, x[2]]
    l(x,a,b) = [x[2], -x[1]]

    h_dyn = Dynam(h, 2, [1,2], [-1, -2])
    k_dyn = Dynam(k, 2, [1,2], [-2, -3])
    l_dyn = Dynam(l, 2, [1,2], [-3, -4])

    cspn = Cospan(FinFunction([1,2,2,3, 3, 4], 4), FinFunction([1,2,3,4], 4))
    d = AlgebraicDynamics.DiscDynam.compose(cspn)(h_dyn, k_dyn, l_dyn)

    input = subpart(d, :value)
    output = zero(input)
    @test update!(output, d, input, 10, 2) == [-10, 18, 18, -4, -4, 3]
    @test get_exposed(d, output) == [-10, 18, -4, 3]
    @test update!(output, d, input, 0, 0) == [0, 2, 2, -4, -4, 3]
    @test get_exposed(d, output) == [0, 2, -4, 3]

end

end #testset
