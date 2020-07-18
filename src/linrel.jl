module LinearRelations

using Catlab
using LinearAlgebra
using LinearMaps
import LinearMaps: UniformScalingMap, BlockDiagonalMap

π₁(A::AbstractMatrix, n::Int) = A[  1:n  , :]
π₂(A::AbstractMatrix, n::Int) = A[n+1:end, :]

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

function compose(f::LinRel,g::LinRel)
    A, B, C, D = f.A, f.B, g.A, g.B
    # definition of nullspace for clarity
    # Σ = svd(hcat(B, C), full=true)
    # n = size(B,2)+size(C,2)
    # nσ⁺ = length(Σ.S)
    # n₀ = n-nσ⁺
    # V₀ = Σ.V[:, n₀+1:end]
    # actually built into julia
    M = Matrix(hcat(B,-C))
    # we need to use an iterative solver for computing the nullspace here
    V₀ = nullspace(M)
    nb = size(B,2)
    @assert norm(Matrix(B*π₁(V₀, nb) - C*π₂(V₀, nb))) < 1e-8
    f, g = A*π₁(V₀, size(B,2)), D*π₂(V₀, size(B,2))
    return LinRel(f,g)
end

inv(f::LinRel) = LinRel(f.B, f.A)
function in(x::Vector, R::LinRel, y::Vector)
    sol = Matrix(vcat(Matrix(R.A), Matrix(R.B)))\vcat(x,y)
    @show sol
    a = norm(sol)/hypot(norm(x),norm(y))
    return a > 1e-8
end

⊕(a,b) = oplus(a,b)
⋅(a,b) = compose(a,b)


# f,g = rand(Float64, 3,2), rand(Float64, 4,2)
A, B = [2 3; -1 4; 1 1], [1 2;]
ΣA, ΣB = svd(A), svd(B)
C, D = [-1 2], [3 -1; 0 1]
ΣC, ΣD = svd(C), svd(D)


h = compose(oplus(LinRel([1 0; 0 1]), LinRel([1 0; 0 1])), LinRel([1 1 1 1]))
@assert inv(LinRel([1 2; 2 1])).A == [1 2; 2 1]
@show inv(LinRel([1 0; 0 1]))
@show inv(LinRel(UniformScalingMap(1, 2))) ⊕ inv(LinRel(UniformScalingMap(1, 2)))

h = compose(oplus(LinRel(LinearMap([1 2; 0 1])), LinRel(UniformScalingMap(1, 2))), LinRel([1 1 1 1]))
@show Matrix(h.A)
@show Matrix(h.B)
hinv = compose(inv(LinRel(LinearMap([1 1 1 1]))), oplus(inv(LinRel(UniformScalingMap(1,2))), inv(LinRel(UniformScalingMap(1,2)))))
hinv = compose(inv(LinRel(LinearMap([1 1 1 1]))), oplus(inv(LinRel(LinearMap([1 2; 0 1]))), inv(LinRel(UniformScalingMap(1,2)))))
@show Matrix(hinv.A)
@show Matrix(hinv.B)

@show in([0.27735, 0.83205, 0.27735, 0.27735], h, [0.27735]) == false
@show in([0.27735], hinv, [0.27735, 0.83205, 0.27735, 0.27735]) == false

end

# h = compose(LinRel(A,B), LinRel(C,D))
# A′, B′ = h.A, h.B
# h

# A, B = [2 3; -1 4; 1 1], [1 2; 3 4]
# ΣA, ΣB = svd(A), svd(B)
# C, D = [-1 2; 3 4], [3 -1; 0 1]
# ΣC, ΣD = svd(C), svd(D)

# # svd(hcat(B, -C)).S
# h = compose(LinRel(A,B), LinRel(C,D))
# A′, B′ = h.A, h.B

# A, B = [2 3; -1 4; 1 1], [1 2; 3 4]
# ΣA, ΣB = svd(A), svd(B)
# C, D = [1 2; 3 4], [3 -1; 0 1]
# ΣC, ΣD = svd(C), svd(D)

# h = compose(LinRel(A,B), LinRel(C,D))
# A′, B′ = h.A, h.B
# f = LinRel(svd(randn(5,4), full=true), svd(randn(3,4), full=true))
# g = LinRel(svd(randn(3,2), full=true), svd(randn(7,2), full=true))
# compose(f, g)
