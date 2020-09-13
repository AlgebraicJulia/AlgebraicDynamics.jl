using LinearAlgebra
# import LinearAlgebra: ⋅
using Catlab
import Catlab.Theories:
  Ob, Hom, dom, codom, compose, ⋅, ∘, id, oplus, ⊕, mzero, swap,
  dagger, dunit, dcounit, mcopy, Δ, delete, ◊, mmerge, ∇, create, □,
  plus, +, zero, coplus, cozero, meet, top, join, bottom,
  proj1, proj2, pair, copair

using Catlab.LinearAlgebra
import Catlab.LinearAlgebra: scalar
using Test
using AlgebraicDynamics.LinRels

@testset "Syntactic" begin
    X,Y,Z,U,V,W = Ob.([ FreeLinearRelations ], [:X,:Y,:Z,:U,:V,:W])
    f = Hom(:f, X, Y)
    g = Hom(:g, Y, Z)
    ud = Hom(:ud, X⊕X⊕X, X)
    h = f⋅g
    @test dom(h) == dom(f)
    @test codom(h) == codom(g)
    F = semantics(Dict(X=>LinRelDom(1),
                       Y=>LinRelDom(4),
                       Z=>LinRelDom(2),
                       U=>LinRelDom(3),
                       f=>LinRel(ones(1,4)/4, I(4)),
                       g=>LinRel([1 0;
                                  1 0;
                                  0 1;
                                  0 1], I(2)),
                       ud=>LinRel(I(3), ones(1,3)/3))
                  )
    @testset "One grid cell Laplacian" begin
        @test F(f).A == ones(1,4)/4
        @test F(f).B == I(4)
        qrf = QRLinRel(F(f))
        @test qrf([0], [1,1,-1,-1])
        @test qrf([1], [1,1,-1,-1]) == false
        @test qrf.r == 4
        @test norm(projker(qrf.QR.Q[:, 1:4], [-1/4, 1,1,-1,-2])) < 1e-12
        @test qrf([0], [1,1,-1,-2]) == false
        @test F(f).B*[1,1,-1,-2] == [1,1,-1,-2]
        @test F(f).A*[1,1,-1,-2] == [-1/4]
        @test norm(projker(qrf.QR.Q[:, 1:4], [-1/4, 1,1,-1,-2])) < 1e-12
        @test qrf([-1/4], [1,1,-1,-2])
    end
    @testset "Boundary Laplacian Relation" begin
        @test F(f⋅g)([0], [1, -1])
        @test F(f⋅g)([-1/2], [1, -2])
        @test F(f⋅g)([0], [2, -2])
        @test F(f⋅g)([2], [2, 2])
        @test F(f⋅g)([-1/4], [1, -2]) == false
        V₀ = LinRels.pullback(F(f),F(g))
        y₁ = F(f).B*π₁(V₀, 4)
        y₂ = F(g).A*π₂(V₀, 4)
        @test norm(y₁ .- y₂) < 1e-12

        one4 = LinRel(Int[], ones(4))
        # one4 = LinRel(Float64[], ones(4))
        @test inv(one4).A == ones(4)

        # @test (F(f)⋅one4)([0], [0,0,0,0])
        @test (F(f)⋅inv(one4))([1], Float64[])

        constbound = F(f)⋅LinRel(ones(4,1), I(1))⋅LinRel(ones(1), Float64[])
        @test constbound([4], Float64[])
        @test constbound([3], Float64[])
        @test (F(f)⋅scalar(F(Y), 4)⋅inv(one4))([4], Float64[])
        # @test (F(f)⋅scalar(F(Y), 4)⋅inv(one4))([3], Float64[])== false
        # @test (F(f)⋅one4)([2], 2ones(4))
        # @show create(F(X))
        # @show F(create(X))

        # @show y₁
        # @show π₁(V₀, 4)
        # @show F(f).A
        # @show F(f).A*π₁(V₀, 4)
        # @show π₂(V₀, 4)
        # @show F(g).B
        # @show F(g).B*π₂(V₀, 4)
    end
    @testset "Upwind Differencing" begin
        @test dom(F(id(X⊕X))) == LinRelDom(2)
        @test codom(F(id(X⊕X))) == LinRelDom(2)
        @test codom(F(create(X⊕X))) == LinRelDom(2)
        @test dom(F(create(X⊕X))) == LinRelDom(0)
        s1 = F(id(X⊕X)⊕create(X⊕X)⊕id(X⊕X))
        @test dom(s1) == LinRelDom(4)
        @test codom(s1) == LinRelDom(6)
        # @show F(create(X⊕X))
        # @show F(mcopy(X⊕X))
        cp = create(X)⋅mcopy(X)⋅mcopy(X⊕X)
        s1 = F(id(X⊕X)⊕(cp)⊕id(X⊕X))
        @test dom(s1) == LinRelDom(4)
        @test codom(s1) == LinRelDom(8)
        s2 = F(ud⊕id(X⊕X)⊕ud)
        @test dom(s2) == LinRelDom(8)
        @test codom(s2) == LinRelDom(4)

        s3 = F(plus(X)⊕plus(X))
        @test dom(s3).n == 4
        @test codom(s3).n == 2

        s = compose(s1,s2,s3)
        @test dom(s3).n == 4
        @test codom(s3).n == 2

        @test s(ones(4), ones(2))
        @test s([-1, 1, 1, -1], ones(2))
        @test s([-1, 1, 1, -1], [1, 2]) == false
        @test s([-1, 1, 1, -1], 2ones(2))

        s² = oplus(F(id(X)), s, F(id(X)))⋅s
        @test QRLinRel(s²).r == 7
        x = [1,0,-1,-1,0,1]
        y = s²(x)
        @test s²(x,y)
        s³ = oplus(F(id(X)), s², F(id(X)))⋅s
        @test QRLinRel(s³).r == 9
        x = [2,1,0,-1,-1,0,1, 2]
        y = s³(x)
        @test s³(x,y)

        @test QRLinRel(F(Δ(X)⊕id(X⊕X)⊕Δ(X))⋅s²).r == 5
        @test QRLinRel(F(Δ(X)⊕id(X⊕X⊕X⊕X)⊕Δ(X))⋅s³).r == 7

        X² = X⊕X
        I₁ = id(X)
        I₂ = id(X²)
        Δ₂ = Δ(X²)
        p₁ = plus(X)
        c₂ = create(X)⋅mcopy(X)
        t₁ = oplus(I₂, c₂⋅Δ₂,I₁,c₂⋅Δ₂,I₂)
        t₂ = oplus(ud, I₂, ud, I₂, ud)
        t₃ = oplus(p₁,(p₁⊕I₁)⋅p₁,p₁)
        tex = compose(t₁,t₂,t₃)

        t = F(tex)
        q = QRLinRel(t)
        @test QRLinRel(t).r == 7
        x = ones(5)
        # @show Q₁(q)'x
        # @show Q₂(q)*Q₁(q)'x
        x₀ = t.A*ones(size(t.A,2))
        y₀ = t.B*ones(size(t.B,2))
        x,y = solution(t, ones(size(t.A,2)))
        @test norm(x₀-x) < 1e-12
        @test norm(y₀-y) < 1e-12
        @test norm(projker(Q₁(q), x))/norm(x) > 1e-12
        @test norm(projker(Q₂(q), y))/norm(y) > 1e-12
        @test norm(projker(vcat(Q₁(q), Q₂(q)), vcat(x,y)))/norm(vcat(x,y)) < 1e-12
        @test t(x,y)

        #solving for y for a fixed x
        ŷ = t(x)
        @test isnothing(ŷ)
        @test_throws MethodError norm(projker(vcat(Q₁(q), Q₂(q)), vcat(x,ŷ)))

        x₀ = Q₁R₁(q)*ones(q.r)
        x₁ = Q₁(q)*Q₁(q)'x₀
        x₂ = x₀ - projker(Q₁(q), x₀)
        x₃ = projker(Q₁(q), x₂)
        # @show t(x₁)
        # @show t(x₂)
        # @show t(x₃)
        ŷ = t(Q₁(q)*Q₁(q)'x₀)
        @test_broken !isnothing(ŷ)
        @test_broken q(x₀,ŷ)

        ŷ = t(Q₁(q)*Q₁(q)'x)
        @test_broken !isnothing(ŷ)
        @test_broken q(x,ŷ)

        t² = F(I₁⊕tex⊕I₁)⋅t
        @test QRLinRel(t²).r == 9
        x,y = solution(t², ones(size(t².A,2)))
        @test t²(x,y)
        x = ones(7)
        @test isnothing(t²(x))
    end
    @testset "Solving for fixed x" begin
        f = LinRel(ones(2,2), ones(1,2))
        q = QRLinRel(f)
        x,y = Q₁(q)*ones(q.r), Q₂(q)*ones(q.r)
        @test f(x,y)
        v = ones(2,1)
        v̂ = v/norm(v)
        g = LinRel(I(1), v̂)
        h = g⋅f
        @test dom(h) == dom(g)
        @test codom(h) == codom(f)
        x̂ = Q₁R₁(QRLinRel(h))*ones(2,1)
        ŷ = Q₂R₁(QRLinRel(h))*ones(2,1)
        @test h(x̂[:,1], ŷ[:,1])
        @test h(x̂[:,1], ŷ[:,1])

        cx = vecrel(x̂[:, 1])
        @test dom(cx).n == 0
        @test codom(cx).n == length(x̂)

        x = f.A*ones(size(f.A,2))
        y = f.B*ones(size(f.B,2))
        @test_broken !isnothing(f(x))
        @test f(x,y)
        f̂ = vecrel(x)⋅f
        @test size(f̂.B) == (1, 2)
        @test f̂([], y)
        @test f̂([], [3])

        @testset "Bigger System" begin
            f = LinRel(I(3), [1 2 3; -1 0 1; 0 1 0])
            x = ones(3)
            y = f.B*ones(3)

            f̂ = vecrel(x)⋅f
            @test size(f̂.A) == (0,1)
            @test size(f̂.B) == (3,1)
            z = LinRels.solve(f, x)
            @test z/z[1] == f̂.B/f̂.B[1]
            @test inrange(f̂.B, y)
            @test !inrange(f̂.B, [-1,0,1])

            @test f̂([], y)
            @test !f̂([], [-1, 0, 1])

            f⁻¹ = inv(f)
            @test inrange(f⁻¹.B, x)
            @test f⁻¹(y, x)
            i¹ = f⋅f⁻¹
            i² = f⁻¹⋅f
            @test dom(i¹) == dom(f)
            @test codom(i¹) == codom(f⁻¹)
            @test dom(i²) == dom(f⁻¹)
            @test codom(i²) == codom(f)
            @test size(i¹.A) == (3,3)
            @test size(i².A) == (3,3)
        end

        # @show compose(f̂, LinRel(Matrix(ŷ), ones(1,2)))
        # @test !isnothing(h([ 1 ]))
    end
end

@testset "Finite Volume Equilibria" begin
    R = LinRelDom(1)
    R² = LinRelDom(2)
    R³ = LinRelDom(3)
    R⁴ = LinRelDom(4)

    X,Y,Z,U,V,W = Ob.([FreeLinearRelations], [:X,:Y,:Z,:U,:V,:W])
    f = Hom(:f, X, Y)
    g = Hom(:g, Y, Z)
    L₄ = Hom(:ud, X⊕X⊕X, X)
    h = f⋅g
    @test dom(h) == dom(f)
    @test codom(h) == codom(g)
    F = semantics(Dict(X=>R,
                       Y=>R²,
                       Z=>R³,
                       U=>R⁴,
                       f=>LinRel(ones(1,3), I(1)),
                       g=>LinRel([1 0;
                                  1 0;
                                  0 1;
                                  0 1], I(2)),
                       )
                  )

    L⁴ = (id(R³)⊕dunit(R))⋅((sum(R⁴)⋅inv(zero(R)))⊕id(R))
    @test L⁴([1,2,3], [6])
    @test L⁴([1,1,1], [3])
    @test L⁴([1,0,1], [2])
    @test !L⁴([1,0,1], [3])
    @test !L⁴([1,1,1], [-3])

    L¹ = (dunit(R)⊕id(R³))⋅(id(R)⊕(sum(R⁴)⋅inv(zero(R))))
    @test L¹([1,2,3], [6])
    @test L¹([1,1,1], [3])
    @test L¹([1,0,1], [2])
    @test !L¹([1,0,1], [3])
    @test !L¹([1,1,1], [-3])

    cell² = L⁴⋅inv(L⁴)
    @test size(cell².A) == (3,9)
    @test size(cell².B) == (3,9)
    @test cell²(ones(3), ones(3))
    ####################
    #                  #
    #    0      0      #
    #    |      |      #
    # x--L⁴--v--L¹--y  #
    #    |      |      #
    #    0      0      #
    #                  #
    ####################
    L¹ = (dunit(R)⊕id(R³))⋅(id(R)⊕(sum(R⁴)⋅inv(zero(R))))
    L⁴ = (id(R³)⊕dunit(R))⋅((sum(R⁴)⋅inv(zero(R)))⊕id(R))
    cell²sealed = (id(R)⊕vecrel([0,0]) )⋅cell²⋅(inv(vecrel([0,0]))⊕id(R))
    @test cell²sealed([1], [1])
    @test !cell²sealed([1], [2])

    @test dom(L¹).n == 3
    @test codom(L¹).n == 1
    pL = (zero(R²)⊕id(R))⋅ L¹
    @test dom(pL).n == 1
    @test codom(pL).n == 1
    chain(f::LinRel, n::Int) = foldl(compose, (f for i in 1:n))
    #############################################
    #                                           #
    #    0       0       0       0       0      #
    #    |       |       |       |       |      #
    # x--L¹--v₁--L¹--v₂--L¹--v₃--L¹--v₄--L¹--y  #
    #    |       |       |       |       |      #
    #    0       0       0       0       0      #
    #                                           #
    #############################################
    pipe = chain(pL, 5)
    @test dom(pipe).n == 1
    @test codom(pipe).n == 1
    @test pipe([1], [1])
    @test !pipe([1], [2])
end

function testcomposite(f::QRLinRel, g::QRLinRel)
    h = compose(f,g)
    w = ones(size(h.QR,2))
    x = Q₁R₁(h)*w
    z = Q₂R₁(h)*w
    @test h(x, z)
    w = randn(size(h.QR,2))
    w = w .- sum(w)
    x = Q₁R₁(h)*w
    z = Q₂R₁(h)*w
    @test h(x, z)

    f̂ = LinRel(f)
    ĝ = LinRel(g)
    V₀ = LinRels.pullback(f̂, ĝ)
    b = size(f̂.B,2)
    p₁, p₂ = π₁(V₀, b), π₂(V₀, b)
    w = ones(size(V₀, 2))
    y₁ = f̂.B*p₁*w
    y₂ = ĝ.A*p₂*w
    @test norm(y₁ - y₂) < 1e-8
end



@testset "Containment" begin
    A = [1 1 0; 1 1 0; 1 2 1]
    QR = qr(A)
    Q, R = QR.Q, QR.R
    r = rankr(R, 1e-12)
    v = A*[1, 2, 3]
    @test norm(projker(Q[:, 1:r],v)) < 1e-12
    vf = copy(v)
    vf[2] *= -1
    @test norm(projker(Q[:, 1:r], [3,-3,8])) > 1e-12
    @test norm(projker(Q[:, 1:r], vf)) > 1e-12
    vf = copy(v)
    vf[2] += 1
    vf[3] += 1
    @test norm(projker(Q[:, 1:r],vf)) > 1e-12

    vf = copy(v)
    vf[3] += 1
    @test norm(projker(Q[:, 1:r],vf)) < 1e-12

    f = QRLinRel(A[1:2, :], A[3:end, :])
    v = A*[1, 2, 3]
    norm(projker(Q[:,1:r], v))
    @test f(v[1:2], v[3:end])
    @test !f([v[1], -v[2]], [0])
    @test !f([v[1], -v[2]], v[3:end])
end

@testset "NullContainment" begin
    A = Matrix(I(5))
    f = QRLinRel(A, 2A)
    @test f(ones(5),   2*ones(5))
    @test !f(-ones(5), 2*ones(5))
    @test !f(ones(5), -2*ones(5))
    @test f(zeros(5),   zeros(5))
    @test !f(zeros(5),   ones(5))
    @test !f(ones(5),   zeros(5))
end

@testset "Composition" begin
    @testset "combo 1" begin
        A = [1 1 0; 1 1 0]
        B = Matrix([1 2 1])
        f = QRLinRel(A,B)
        g = QRLinRel(B,A)
        h = compose(f,g)
        @test size(h.QR.Q) == (4,4)
        @test h.n == 2
        @test h.m == 2
        @test size(R₁(h)) == (2,5)
        testcomposite(f,g)
    end
    @testset "combo 2" begin
        A = [1 1 0; 1 1 0]
        B = Matrix([1 2 1; 2 3 1])
        f = QRLinRel(A,B)
        g = QRLinRel(B,A)
        testcomposite(f,g)
    end
    @testset "combo 3" begin
        A = [1 1 0; -1 1 0]
        B = Matrix([1 2 1; 2 3 1])
        f = QRLinRel(A,B)
        g = QRLinRel(B,A)
        testcomposite(f,g)
    end
    @testset "combo 4" begin
        A = [1 1 0; -1 1 0]
        B = Matrix([1 2 1; 2 3 1])
        C = Matrix([0 2 1; -1 3 1])
        f = QRLinRel(A,B)
        g = QRLinRel(C,A)
        testcomposite(f,g)
        D = [1 1 0; -1 1 -1]
        g = QRLinRel(C,D)
        testcomposite(f,g)
    end
end

@testset "LinearFunctions Instance" begin
    f = LinRel(ones(3,3), ones(4,3))

    @test dom(f).n == 3
    @test codom(f).n == 4
    QRf = QRLinRel(f)
    @test QRf(f.A*ones(3), f.B*ones(3))
    @test !QRf(f.A*ones(3)+[0,0,1], f.B*ones(3))

    I5 = id(LinRelDom(5))
    @test dom(I5).n ==5
    @test codom(I5).n ==5
    QRI5 = QRLinRel(I5)
    @test QRI5(ones(5), ones(5))
    @test !QRI5(2ones(5), ones(5))
    @test !QRI5(ones(5), 2ones(5))
    @test QRI5(2ones(5), 2ones(5))

    @test size(oplus(ones(3,3), ones(2,2))) == (5,5)

    @test oplus(LinRelDom(5), LinRelDom(3)).n == 8
    foplusI2 = oplus(f,id(LinRelDom(2)))
    QRfoplusI2 = QRLinRel(foplusI2)

    x = vcat(f.A*ones(3), ones(2))
    y = vcat(f.B*ones(3), ones(2))
    @test QRfoplusI2(x,y)
    @test QRLinRel(foplusI2 ⋅ id(LinRelDom(6)))(x,y)

    @test QRLinRel(I5⋅foplusI2)(x,y)

    V₁ = LinRelDom(1)
    V₂ = LinRelDom(2)
    V₃ = LinRelDom(3)

    @test mzero(LinRelDom) == LinRelDom(0)
    @testset "Swap" begin
        @test  swap(V₁, V₁)([1,2], [2,1])
        @test !swap(V₁, V₁)([1,2], [1,2])
        @test !swap(V₁, V₁)([1,2], [1,3])
        @test  swap(V₂, V₃)([1,2,3,4,5], [3,4,5,1,2])
        @test !swap(V₂, V₃)([1,2,3,4,5], [4,4,5,1,2])
        @test !swap(V₃, V₂)([1,2,3,4,5], [3,4,5,1,2])
    end

    @testset "Mcopy" begin
        @test  mcopy(LinRelDom(1))([ 1], [1,1])
        @test !mcopy(LinRelDom(1))([-1], [1,1])
        @test  mcopy(LinRelDom(2))([ 1, 2], [ 1,2,1,2])
        @test !mcopy(LinRelDom(2))([-1, 2], [-1,2,1,2])
    end

    @testset "Delete" begin
        @test delete(V₁)([1], [])
        @test delete(V₂)([1, 2], [])
        @test_throws Exception delete(V₂)([1, 2], [2])
    end

    @testset "Plus" begin
        @test  plus(V₁)([1,2], [3])
        @test !plus(V₁)([1,3], [3])
        @test  plus(V₂)([1,2,3,4], [4,6])
        @test !plus(V₂)([1,3,3,4], [4,6])
    end

    @testset "Zero" begin
        for i in 1:4
            z = zero(LinRelDom(i))
            @test z([], zeros(i))
            @test norm((z⋅id(LinRelDom(i))).B) < 1e-14
        end
    end

    @testset "Scalar/Antipode" begin
        for i in 1:4
            V = LinRelDom(i)
            @test scalar(V, 2)(ones(i), 2ones(i))
            @test antipode(V)(ones(i), -ones(i))
            @test ( scalar(V,2)⋅antipode(V) )(ones(i), -2ones(i))
            @test ( antipode(V)⋅scalar(V,2) )(ones(i), -2ones(i))
            @test ( antipode(V)⋅scalar(V,2)⋅antipode(V))(ones(i), 2ones(i))
        end
    end

    @testset "Pair/Copair" begin
        V = LinRelDom(2)
        f = scalar(V, 2)
        g = scalar(V, 3)
        @test  pair(f,g)([1,2], [2,4,3,6])
        @test !pair(f,g)([1,2], [3,4,3,6])

        f = scalar(V, 2)
        g = scalar(V, 3)
        @test  copair(f,g)([1,2,3,4], [11,16])
        @test !copair(f,g)([1,2,3,4], [12,16])
    end

    @testset "Proj/Coproj" begin
        V = LinRelDom(2)
        W = LinRelDom(4)
        f = scalar(V, 2)
        g = scalar(V, 3)
        f̂ = (f⊕g)⋅proj1(V,V)
        ĝ = (f⊕g)⋅proj2(V,V)

        @test f̂([1,2, 1, 1], [2,4])
        @test ĝ([1,2, 1, 1], [3,3])

    end

    @testset "Caps and Cups" begin
        @test dunit(LinRelDom(1))([], [1, -1])
        @test !dunit(LinRelDom(1))([], [1, -2])
        @test dunit(LinRelDom(2))([], [1, 2, -1, -2])
        @test !dunit(LinRelDom(2))([], [1, 2, -2, 1])

        @test dcounit(LinRelDom(1))([1, -1], [])
        @test !dcounit(LinRelDom(1))([1, -2], [])
        @test dcounit(LinRelDom(2))([1, 2, -1, -2], [])
        @test !dcounit(LinRelDom(2))([1, 2, -2, 1], [])
    end

    # inv(f::LinRel) = LinRel(f.B, f.A)

end

