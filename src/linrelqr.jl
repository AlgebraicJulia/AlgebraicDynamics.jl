module LinRels

import Base: +
using LinearAlgebra
# import LinearAlgebra: ⋅
using Catlab
import Catlab.Theories:
  Ob, Hom, dom, codom, compose, ⋅, ∘, id, oplus, ⊕, mzero, swap,
  dagger, dunit, dcounit, mcopy, Δ, delete, ◊, mmerge, ∇, create, □,
  plus, +, zero, coplus, cozero, meet, top, join, bottom
using Catlab.LinearAlgebra
using Test

⊕(a,b) = oplus(a,b)
⋅(a,b) = compose(a,b)

π₁(A::AbstractMatrix, n::Int) = A[  1:n  , :]
π₂(A::AbstractMatrix, n::Int) = A[n+1:end, :]

rankr(R, tol=1e-15) = begin
    nrms = norm.(R[i,:] for i in 1:size(R)[1])
    p(x) = x > tol
    sum(p.(nrms))
end

projker(Q, v) = v-Q*Q'v

struct LinRelDom
    n::Int
end

struct LinRel{T,U}
    A::T
    B::U
end

# function LinRel(A::Matrix,B::Matrix)
#     size(A,2) == size(B,2) || error("Span construction invalide")
#     return new(A,B)
# end


# LinRel(B) = LinRel(UniformScalingMap(1, size(B,2)), B)


@instance LinearFunctions(LinRelDom, LinRel) begin
    dom(f::LinRel) = LinRelDom(size(f.A,1))
    codom(f::LinRel) = LinRelDom(size(f.B,1))
    id(X::LinRelDom) = LinRel(I(X.n), I(X.n))
    oplus(X::LinRelDom,Y::LinRelDom) = LinRelDom(X.n + Y.n)
    oplus(f::LinRel,g::LinRel) = LinRel(f.A⊕g.A, f.B⊕g.B)
    oplus(A::AbstractMatrix, B::AbstractMatrix) = oplus(Matrix(A), Matrix(B))
    oplus(A::Matrix,B::Matrix) = hvcat((2,2),
                                       A, zeros(size(A,1), size(B,2)),
                                       zeros(size(B,1), size(A,2)), B)
    # oplus(a...) = BlockDiagonalMap(a...)
    oplus(a::Diagonal,b::Diagonal) = Diagonal(vcat(a.diag, b.diag))

    mzero(::Type{LinRelDom}) = LinRelDom(0)
    swap(X::LinRelDom, Y::LinRelDom) = LinRel(I(X.n+Y.n), hvcat((2,2), zeros(Y.n,X.n), I(Y.n), I(X.n), zeros(X.n,Y.n)))
    braid(X::LinRelDom, Y::LinRelDom) = LinRel(I(X.n+Y.n), hvcat((2,2), zeros(Y.n,X.n), I(Y.n), I(X.n), zeros(X.n,Y.n)))

    mcopy(X::LinRelDom) = LinRel(I(X.n), vcat(I(X.n), I(X.n)))
    delete(X::LinRelDom) = LinRel(I(X.n), zeros(0, X.n))
    plus(X::LinRelDom) = LinRel(I(2*X.n), hcat(I(X.n), I(X.n)))
    zero(X::LinRelDom) = LinRel(ones(0,1), zeros(X.n, 1))

    scalar(X::LinRelDom, c::Number) = LinRel(I(X.n), c*I(X.n))
    antipode(X::LinRelDom) = LinRel(I(X.n), -I(X.n))

    pair(f::LinRel, g::LinRel) = mcopy(dom(f)) ⋅ (f ⊕ g)
    copair(f::LinRel, g::LinRel) = (f ⊕ g) ⋅ plus(codom(f))
    plus(f::LinRel, g::LinRel) = mcopy(dom(f)) ⋅ (f⊕g) ⋅ plus(codom(f))

    proj1(A::LinRelDom, B::LinRelDom) = id(A) ⊕ delete(B)
    proj2(A::LinRelDom, B::LinRelDom) = delete(A) ⊕ id(B)
    coproj1(A::LinRelDom, B::LinRelDom) = id(A) ⊕ zero(B)
    coproj2(A::LinRelDom, B::LinRelDom) = zero(A) ⊕ id(B)
    compose(f::LinRel,g::LinRel) = begin
        codom(f) == dom(g) || error("Dimension Mismatch $(codom(f).n)!=$(dom(g).n)")
        A, B, C, D = f.A, f.B, g.A, g.B
        M = Matrix(hcat(B,-C))
        # we need to use an iterative solver for computing the nullspace here
        V₀ = nullspace(M)
        nb = size(B,2)
        @assert norm(Matrix(B*π₁(V₀, nb) - C*π₂(V₀, nb))) < 1e-8
        f, g = A*π₁(V₀, size(B,2)), D*π₂(V₀, size(B,2))
        return LinRel(f,g)
    end
    adjoint(f::LinRel) = LinRel(adjoint(f.A), adjoint(f.B))
end

create(X::LinRelDom) = LinRel(zeros(0,1), ones(X.n,1))
inv(f::LinRel) = LinRel(f.B, f.A)





function pullback(f::LinRel,g::LinRel)
    A, B, C, D = f.A, f.B, g.A, g.B
    M = Matrix(hcat(B,-C))
    # we need to use an iterative solver for computing the nullspace here
    V₀ = nullspace(M)
    nb = size(B,2)
    @assert norm(Matrix(B*π₁(V₀, nb) - C*π₂(V₀, nb))) < 1e-8
    f, g = A*π₁(V₀, size(B,2)), D*π₂(V₀, size(B,2))
    return V₀
end

struct QRLinRel{T}
    n::Int
    m::Int
    r::Int
    QR::T
end

function QRLinRel(A::Matrix, B::Matrix)
    M = vcat(A, B)
    QR = qr(M, Val(true))
    r = rankr(QR.R, 1e-12)
    return QRLinRel(size(A,1), size(B,1), r, QR)
end

QRLinRel(f::LinRel) = QRLinRel(Matrix(f.A), Matrix(f.B))


Q₁(f::QRLinRel) = f.QR.Q[1:f.n, 1:f.r]
Q₂(f::QRLinRel) = f.QR.Q[(f.n+1):end, 1:f.r]
R₁(f::QRLinRel) = f.QR.R[1:f.r, :]
Q₁R₁(f::QRLinRel) = Q₁(f)*R₁(f)
Q₂R₁(f::QRLinRel) = Q₂(f)*R₁(f)

LinRel(f::QRLinRel) = LinRel(Q₁R₁(f), Q₂R₁(f))

dom(f::QRLinRel) = LinRelDom(f.n)
codom(f::QRLinRel) = LinRelDom(f.m)

# id(X::LinRelDom) = LinRel(LinRel(I(X.n), I(X.n)))
oplus(f::QRLinRel,g::QRLinRel) = QRLinRel(Q₁R₁(f)⊕Q₁R₁(g), Q₂R₁(f)⊕Q₂R₁(g))

function compose(f::QRLinRel, g::QRLinRel)
    h = compose(LinRel(f), LinRel(g))
    QRLinRel(h.A, h.B)
end

function in(x::Vector, f::QRLinRel, y::Vector, tol=1e-12)
    norm(vcat(x,y)) > tol || return true
    # r = projker(f.QR.Q, vcat(x,y))
    z = vcat(x,y)
    r = projker(f.QR.Q[:, 1:f.r], z)
    return norm(r)/norm(z) < tol
end

(f::QRLinRel)(x::Vector, y::Vector) = in(x,f,y)
(f::QRLinRel)(x::Vector) = Q₂(f)*Q₁(f)'x
(f::LinRel)(x::Vector, y::Vector) = QRLinRel(f)(x,y)
(f::LinRel)(x::Vector) = QRLinRel(f)(x)

function semantics(generators::AbstractDict=Dict(), terms::AbstractDict=Dict())
    return ex->functor((LinRelDom, LinRel),
                       ex,
                       generators=generators,
                       terms=terms)
end

@testset "Syntactic" begin
    X,Y,Z,U,V,W = Ob.([ FreeLinearRelations ], [:X,:Y,:Z,:U,:V,:W])
    f = Hom(:f, X, Y)
    g = Hom(:g, Y, Z)
    ud = Hom(:ud, U, X)
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
        V₀ = pullback(F(f),F(g))
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
        cp = create(X⊕X)⋅mcopy(X⊕X)
        # @show F(cp)
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
        x = [1,0,-1,-1,0,1]
        y = s²(x)
        @test s²(x,y)
    end

end

function testcomposite(f::QRLinRel, g::QRLinRel)
    h = compose(f,g)
    w = ones(size(h.QR,2))
    x = Q₁R₁(h)*w
    z = Q₂R₁(h)*w
    @test in(x,h, z)
    w = randn(size(h.QR,2))
    w = w .- sum(w)
    x = Q₁R₁(h)*w
    z = Q₂R₁(h)*w
    @test in(x,h, z)

    f̂ = LinRel(f)
    ĝ = LinRel(g)
    V₀ = pullback(f̂, ĝ)
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
    @test in(v[1:2], f, v[3:end])
    @test !in([v[1], -v[2]], f, [0])
    @test !in([v[1], -v[2]], f, v[3:end])
end

@testset "NullContainment" begin
    A = Matrix(I(5))
    f = QRLinRel(A, 2A)
    @test in(ones(5),   f,  2*ones(5))
    @test !in(-ones(5), f,  2*ones(5))
    @test !in(ones(5),  f, -2*ones(5))
    @test in(zeros(5),  f,   zeros(5))
    @test !in(zeros(5), f,    ones(5))
    @test !in(ones(5),  f,   zeros(5))
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

    # inv(f::LinRel) = LinRel(f.B, f.A)

end


end
