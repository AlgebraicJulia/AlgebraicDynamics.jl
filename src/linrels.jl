module LinRels

import Base: +, sum, inv
using LinearAlgebra
# import LinearAlgebra: ⋅
using Catlab
import Catlab.Theories:
  Ob, Hom, dom, codom, compose, ⋅, ∘, id, oplus, ⊕, mzero, swap,
  dagger, dunit, dcounit, mcopy, Δ, delete, ◊, mmerge, ∇, create, □,
  plus, +, zero, coplus, cozero, meet, top, join, bottom,
  proj1, proj2, pair, copair
using Catlab.LinearAlgebra
import Catlab.LinearAlgebra: scalar, antipode

export LinRelDom, LinRel, vecrel, ncopy, inv,
    solve, solution, semantics, inrange,
    π₁, π₂, rankr, projker, QRLinRel, pullback,
    Q₁, Q₂, Q₁R₁, Q₂R₁, R₁

⊕(a,b) = oplus(a,b)
⋅(a,b) = compose(a,b)


"""    π₁(A::AbstractMatrix, n::Int)

project a matrix M:Rˣ→Rʸ onto the first n dimensions of Rʸ.
"""
π₁(A::AbstractMatrix, n::Int) = A[  1:n  , :]

"""    π₂(A::AbstractMatrix, n::Int)

project a matrix M:Rˣ→Rʸ onto the second y-n dimensions of Rʸ.
"""
π₂(A::AbstractMatrix, n::Int) = A[n+1:end, :]

"""    rankr(R, tol)

estimate the rank of the upper triangular matrix R. It is assumed that R was produced by a QR decomposition with partial pivoting ie. `qr(A Val(true)).R`.
"""
rankr(R, tol=1e-15) = begin
    nrms = norm.(R[i,:] for i in 1:size(R)[1])
    p(x) = x > tol
    sum(p.(nrms))
end

"""    projker(Q,v)

subtract from v, its projection onto the range of Q. Assumes that Q is orthogonal, Q'Q = I.
"""
projker(Q, v) = v-Q*Q'v

"""    LinRelDom

the domain of a linear relation. Represents vector space Rⁿ.
"""
struct LinRelDom
    n::Int
end

@doc raw"""    LinRel

a linear relation stored as a span A,B. Represents the set of pairs of vectors (x,y):

```math
x = Av, y = Bv
```

LinearRelations form a hypergraph category and are an instance of a Bicategory of relations.
"""
struct LinRel{T,U}
    A::T
    B::U
end

# function LinRel(A::Matrix,B::Matrix)
#     size(A,2) == size(B,2) || error("Span construction invalide")
#     return new(A,B)
# end


# LinRel(B) = LinRel(UniformScalingMap(1, size(B,2)), B)


inv(f::LinRel) = LinRel(f.B, f.A)
@instance LinearRelations(LinRelDom, LinRel) begin
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
    mmerge(X::LinRelDom) = inv(mcopy(X))
    delete(X::LinRelDom) = LinRel(I(X.n), zeros(0, X.n))
    plus(X::LinRelDom) = LinRel(I(2*X.n), hcat(I(X.n), I(X.n)))
    zero(X::LinRelDom) = LinRel(ones(0,1), zeros(X.n, 1))

    meet(R::LinRel, S::LinRel) = compose(mcopy(dom(R)), oplus(R,S), mmerge(codom(R)))
    top(A::LinRelDom, B::LinRelDom) = compose(delete(A), create(B))
    join(R::LinRel, S::LinRel) = compose(coplus(dom(R)), oplus(R,S), plus(codom(R)))
    bottom(A::LinRelDom, B::LinRelDom) = compose(cozero(A), zero(B))

    scalar(X::LinRelDom, c::Number) = LinRel(I(X.n), c*I(X.n))
    antipode(X::LinRelDom) = LinRel(I(X.n), -I(X.n))

    create(X::LinRelDom) = LinRel(zeros(0,1), ones(X.n,1))
    dunit(X::LinRelDom) = zero(X)⋅inv(plus(X))
    dcounit(X::LinRelDom) = plus(X)⋅inv(zero(X))

    plus(f::LinRel, g::LinRel) = mcopy(dom(f)) ⋅ (f⊕g) ⋅ plus(codom(f))

    dagger(f::LinRel) = inv(f)
    coplus(X::LinRelDom) = inv(plus(X))

    cozero(X::LinRelDom) = inv(zero(X))

    proj1(A::LinRelDom, B::LinRelDom) = id(A) ⊕ delete(B)
    proj2(A::LinRelDom, B::LinRelDom) = delete(A) ⊕ id(B)
    coproj1(A::LinRelDom, B::LinRelDom) = id(A) ⊕ zero(B)
    coproj2(A::LinRelDom, B::LinRelDom) = zero(A) ⊕ id(B)
    compose(f::LinRel,g::LinRel) = begin
        codom(f) == dom(g) || error("Dimension Mismatch $(codom(f).n)!=$(dom(g).n)")
        A, B, C, D = f.A, f.B, g.A, g.B
        M = Matrix(hcat(Float64.(B),-Float64.(C)))
        # we need to use an iterative solver for computing the nullspace here
        V₀ = nullspace(M)
        nb = size(B,2)
        @assert norm(Matrix(B*π₁(V₀, nb) - C*π₂(V₀, nb))) < 1e-8
        f, g = A*π₁(V₀, size(B,2)), D*π₂(V₀, size(B,2))
        return LinRel(f,g)
    end
    adjoint(f::LinRel) = LinRel(adjoint(f.A), adjoint(f.B))
end

pair(f::LinRel, g::LinRel) = mcopy(dom(f)) ⋅ (f ⊕ g)
proj1(A::LinRelDom, B::LinRelDom) = id(A) ⊕ delete(B)
proj2(A::LinRelDom, B::LinRelDom) = delete(A) ⊕ id(B)
copair(f::LinRel, g::LinRel) = (f ⊕ g) ⋅ plus(codom(f))


"""    sum(X::LinRelDom)

the linear relation representing v↦sum(v).
"""
sum(X::LinRelDom) = LinRel(I(X.n), ones(X.n)')

"""    ncopy(n::Int)

the linear relation representing the nfold copy of a scalar
"""
ncopy(n::Int) = LinRel(I(1), ones(n,1))

# TODO: fix this
# ncopy(X::LinRelDom, n::Int) = LinRel(I(X.n), ones(n,X.n))

"""    vecrel(v)

the linear relation containing the span of one vector as the codomain.
"""
vecrel(v) = begin
    n = length(v)
    compose(compose(create(LinRelDom(1)), ncopy(n)), LinRel(I(n), diagm(v)))
end


"""    pullback(f::LinRel, g::LinRel)

computes the pullback of `X<-f.A- V₁ -f.B-> Y <-g.A- V₂ -g.B-> Z`.
"""
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

"""    LinRel

a linear relation stored as a QR decompositon of

[x]  =  [A][v]
[y]     [B]

Useful for checking correctness of LinRel implementation.
"""
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

"""    in(x::Vector, f::QRLinRel, y::Vector, tol=1e-12)

test if (x,y) in relation f to a fixed relative tolerance.
"""
function in(x::Vector, f::QRLinRel, y::Vector, tol=1e-12)
    norm(vcat(x,y)) > tol || return true
    # r = projker(f.QR.Q, vcat(x,y))
    z = vcat(x,y)
    r = projker(f.QR.Q[:, 1:f.r], z)
    return norm(r)/norm(z) < tol
end

"""    (f::QRLinRel)(x::Vector, y::Vector, tol=1e-12)

test if (x,y) in relation f to a fixed relative tolerance.
"""
(f::QRLinRel)(x::Vector, y::Vector) = in(x,f,y)
# TODO: this is wrong
(f::QRLinRel)(x::Vector, rtol::Real=1e-12) = begin
    norm(projker(Q₁(f), x))/norm(x) < rtol || return nothing
    return Q₂(f)*Q₁(f)'x
end

"""    (f::LinRel)(x::Vector, y::Vector, tol=1e-12)

test if (x,y) in relation f to a fixed relative tolerance.
"""
(f::LinRel)(x::Vector, y::Vector) = QRLinRel(f)(x,y)
(f::LinRel)(x::Vector) = QRLinRel(f)(x)
# (f::QRLinRel)(x::Vector) = Q₂(f)*R₁(f)*(R₁(f) \ (Q₁(f)'x))


"""    solve(f::LinRel, x::Vector)

compute a basis of the solution space of {y | f(x,y)}
"""
function solve(f::LinRel, x::Vector, tol=1e-12)
    f̂ = vecrel(x)⋅f
    Q,R,π = qr(f̂.B, Val(true))
    r = rankr(R, tol)
    Q₁ = Q[:, 1:r]
end

"""    inrange(f::Matrix, y::Vector)

test if a vector is in the range of a matrix using linear solving
"""
inrange(A::AbstractMatrix, y::AbstractVector, tol=1e-12) = norm(y - A*(A\y)) < tol



"""    solution(f::LinRel, v::Vector)

compute the (x,y) pair indexed by v. That is:

    (f.A*v, f.B*v)

"""
solution(f::LinRel, v::Vector) = (f.A*v, f.B*v)
solution(f::QRLinRel, v::Vector) = (Q₁R₁(f)*v, Q₂R₁(f)*v)

"""    semantics(generators::AbstractDict=Dict(), terms::AbstractDict=Dict())

capture a map of generators and terms into a closure that accepts expressions
in a hypergraph category and returns the corresponding LinRel by plugging in
the LinRels for the generators and following the functorial definition.
"""
function semantics(generators::AbstractDict=Dict(), terms::AbstractDict=Dict())
    return ex->functor((LinRelDom, LinRel),
                       ex,
                       generators=generators,
                       terms=terms)
end


end
