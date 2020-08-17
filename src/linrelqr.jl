module LinearRelations

using Catlab
using LinearAlgebra
using LinearMaps
import LinearMaps: UniformScalingMap, BlockDiagonalMap
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

LinRel(B) = LinRel(UniformScalingMap(1, size(B,2)), B)

dom(f::LinRel) = size(f.A,1)
codom(f::LinRel) = size(f.B,1)
id(X::LinRelDom) = LinRel(I(X.n), I(X.n))
oplus(f::LinRel,g::LinRel) = LinRel(f.A⊕g.A, f.B⊗g.B)
oplus(a...) = BlockDiagonalMap(a...)
oplus(a::Diagonal,b::Diagonal) = Diagonal(vcat(a.diag, b.diag))
inv(f::LinRel) = LinRel(f.B, f.A)

function compose(f::LinRel,g::LinRel)
    A, B, C, D = f.A, f.B, g.A, g.B
    M = Matrix(hcat(B,-C))
    # we need to use an iterative solver for computing the nullspace here
    V₀ = nullspace(M)
    nb = size(B,2)
    @assert norm(Matrix(B*π₁(V₀, nb) - C*π₂(V₀, nb))) < 1e-8
    f, g = A*π₁(V₀, size(B,2)), D*π₂(V₀, size(B,2))
    return LinRel(f,g)
end

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
    QR = qr(M)
    r = rankr(QR.R, 1e-12)
    return QRLinRel(size(A,1), size(B,1), r, QR)
end


Q₁(f::QRLinRel) = f.QR.Q[1:f.n, 1:f.r]
Q₂(f::QRLinRel) = f.QR.Q[(f.n+1):end, 1:f.r]
R₁(f::QRLinRel) = f.QR.R[1:f.r, :]
Q₁R₁(f::QRLinRel) = Q₁(f)*R₁(f)
Q₂R₁(f::QRLinRel) = Q₂(f)*R₁(f)

LinRel(f::QRLinRel) = LinRel(Q₁R₁(f), Q₂R₁(f))

dom(f::QRLinRel) = LinRelDom(f.n)
codom(f::QRLinRel) = LinRelDom(f.m)

id(X::LinRelDom) = QRLinRel(I(X.n), I(X.n))
oplus(f::QRLinRel,g::QRLinRel) = QRLinRel(Q₁R₁(f)⊕Q₁R₁(g), Q₂R₁(f)⊕Q₂R₁(g))

function in(x::Vector, f::QRLinRel, y::Vector, tol=1e-12)
    # norm(vcat(x,y)) > tol || return true
    r = projker(f.QR.Q[:, 1:f.r], vcat(x,y))
    return norm(r) < tol
end

function compose(f::QRLinRel, g::QRLinRel)
    h = compose(LinRel(f), LinRel(g))
    QRLinRel(h.A, h.B)
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
        y₂ = ĝ.A*p₁*w
        @test norm(y₁ - y₂) < 1e-8
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
end


@testset "Containment" begin
    A = [1 1 0; 1 1 0; 1 2 1]
    QR = qr(A)
    Q, R = QR.Q, QR.R
    @show Q
    @show R
    @show r = rankr(R, 1e-12)
    v = A*[1, 2, 3]
    @show v
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
    @show norm(projker(Q[:,1:r], v))
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

end
