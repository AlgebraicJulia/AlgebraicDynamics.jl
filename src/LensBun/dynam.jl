using Catlab
using Catlab.CategoricalAlgebra
using Catlab.Theories
import Catlab.Theories: id, dom, codom, compose

"""    VectorField{F}

A dynamical system over a Euclidean Space on `n` dimensions. 
Use `VectorField{Function}` to disable specialization on the vector field function.
"""
struct VectorField{F}
    domain::FinSet # Rⁿ
    field::F  # maps Rⁿ to Tangents(Rⁿ)
end

domain(U::VectorField) = U.domain
tanspace(U::VectorField) = U.domain
vfield(U::VectorField) = U.field

(U::VectorField)(u) = begin
    FinSet(length(u)) == domain(U) || error("Domain Mismatch: state vector has length $(length(u)) was expecting $(domain(U))")
    vfield(U)(u)
end

"""    VectorFieldHom

The structure for storing a homomorphism between VectorField objects. The key axiom is that

    restrict(f, simulate(f, u) - f.u(u)) == 0:domain(f.v)

this means that you can see a version of v inside u by applying the `Pullback(f)` map to restrict states of u to a state of v.
Then apply the vector field for v to get a tangent vector for v. Then you can apply the `Pushforward(f)` map to get a tangent vector for u.

    v::VectorField
    u::VectorField
    f::FinFunction
   
    Pullback(f):    domain(u) → domain(v)
    Pushforward(f): tan(v) → tan(u)

"""
struct VectorFieldHom
    v::VectorField
    u::VectorField
    f::FinFunction
end

dom(ϕ::VectorFieldHom) = ϕ.v
codom(ϕ::VectorFieldHom) = ϕ.u

id(V::VectorField) = VectorFieldHom(V, id(domain(V)))
compose(ϕ::VectorFieldHom, γ::VectorFieldHom) = begin
    VectorFieldHom(dom(ϕ),
                codom(γ),
                ϕ.f ⋅ γ.f)
end

pullback(f::FinFunction) = u -> u[collect(f)]
pushforward(f::FinFunction) = u̇ -> map(codom(f)) do i
    sum(u̇[j] for j in preimage(f, i);init=0.0)
end

""" Pullback{F} wraps F in a struct to make a callable for the action of pulling a function back along `f::F`. 
For `F <: FinFunction`, f: N → M sends Xᴹ to Xᴺ by precomposition.
"""
struct Pullback{F}
    f::F
end

(fꜛ::Pullback{F})(u) where F <: FinFunction = u[collect(fꜛ.f)]

""" Pushforward{F} wraps F in a struct to make a callable for the action of pushing a function forward along `f::F`. 
For `F <: FinFunction`, f: N → M sends Xᴺ to Xᴹ by adding over preimages. 
Requires that X be a commutative additive monoid. The method `zero∘eltype(u::Xᴹ)` should return the unit and sum, should use the addition operator. 
"""
struct Pushforward{F}
    f::F
end

(fꜜ::Pushforward{F})(u̇) where F <: FinFunction = map(codom(fꜜ.f)) do i
    sum(u̇[j] for j in preimage(fꜜ.f, i);init=zero(eltype(u̇)))
end

VectorFieldHom(V, U, f::FinFunction) = begin
    domain(V) == dom(f) || error("FinFunctions induce VectorFieldHoms covariantly")
    domain(U) == codom(f)    || error("FinFunctions induce VectorFieldHoms covariantly")
    return VectorFieldHom(V,U, f)
end

VectorFieldHom(V, f::FinFunction) = begin
    U = VectorField(codom(f), Pushforward(f)∘vfield(V)∘Pullback(f))
    VectorFieldHom(V, U, f)
end

"""    Dynam

The dynamical systems functor D: FinSet → Set that sends finsets `n` to VectorFields on `n` dimensions and
sends finfunctions `f: n → m` to the function that sends VectorFields to VectorFields by defining the morphism defined in `simulate`.
VectorFieldHoms are the morphisms in the category of elements of this functor.
"""
Dynam(X::FinSet) = VectorField # would like dependent type to depend on Int here.

Dynam(f::FinFunction) = (V::VectorField) -> begin
    domain(V) == dom(f) || error("V is not in domain of Dynam(f)")
    codom(VectorFieldHom(dom(f), f))
end

proj(V::VectorField) = domain(V)
proj(f::VectorFieldHom) = f.f


"""    restrict(f::VectorFieldHom, v::AbstractVector)

Apply f.f_state to send states/tangents in `domain(codom(f))` to states/tangents in `domain(dom(f))`.
Uses the fact that the state space of a VectorField is a Euclidean Space to treat states and tangents as vectors. 
"""
restrict(f::VectorFieldHom, u::AbstractVector) = Pullback(proj(f))(u)


"""    pushforward(f::VectorFieldHom, v::AbstractVector)

Apply f.f_tangent to send tangent vectors over `domain(dom(f))` to tangent vectors over `domain(codom(f))`.
"""
pushforward(f::VectorFieldHom, v::AbstractVector) = Pushforward(proj(f))(v)

"""    simulate(f::VectorFieldHom, v::AbstractVector)

Uses f.u to simulate f.v by pulling back the state and pushing forward the tangents.

This name is confusing in the context of numerical simulation.

axiom: simulate(f, v) == f.v(v) 
"""
simulate(f::VectorFieldHom, u::AbstractVector) = pushforward(f, f.v(restrict(f, u)))



using Test
X = FinSet(3)
Y = FinSet(2)
f = FinFunction([1,2,2], X, Y)

v = VectorField(X, x->[x[1] - x[2], x[2]])
u = VectorField(Y, y -> [y[1], -y[2]])

@test proj(v) == X
@test proj(u) == Y

# ϕ = VectorFieldHom(u, v, y->y[[1,2,2]], ẋ->[ẋ[1], ẋ[2]+ẋ[3]])


LV(α,β,γ,δ) = VectorField(FinSet(2),
    u -> [α*u[1] - β*u[1]*u[2],
          γ*u[1]*u[2] + δ*u[2]])

lvact(α,β,γ,δ, f) = VectorFieldHom(LV(α, β, γ, δ), f)

@testset "Inclusion Map2" begin
    f = lvact(1,0.5,0.3,-0.2, FinFunction([1,2], 3))
    for i in 1:10
        r = rand(domain(codom(f)).n)
        # @show simulate(f, r)
        # @show domain(f.v)
        # @show domain(f.u)
        # @show f.u(r)
        # @show f.v(restrict(f, r))
        # @show typeof(restrict(f, r))
        # @show pushforward(f, f.v(restrict(f, r))) - f.u(r)
        # @show simulate(f, r)
        @test proj(f) == FinFunction([1,2], 3)
        @test all(simulate(f, r) - codom(f)(r) .<= 1e-4)
    end
end

@testset "Surj Map2" begin
    f = lvact(1,0.5,0.3,-0.2, FinFunction([1,1], 1))
    for i in 1:10
        r = rand(domain(codom(f)).n)
        @test proj(f) == FinFunction([1,1], 1)
        @test all(simulate(f, r) - codom(f)(r) .<= 1e-4)
    end
end

@testset "Compose" begin
    V = LV(1, 0.5, 0.3, -0.2)

    f = FinFunction([1,1], 3)
    g = FinFunction([1,2,1], 2)
    h = compose(f,g)

    ϕ = VectorFieldHom(V, f)
    U = codom(ϕ)
    
    γ = VectorFieldHom(U, g)
    W = codom(γ)
    η = compose(ϕ,γ)

    @test dom(η) == dom(ϕ)
    @test codom(η) == codom(γ)
    @test collect(η.f) == collect(h)

    for i in 1:10
        r = rand(domain(codom(η)).n)
        @test collect(proj(η)) == collect(h)
        @test all(simulate(η, r) - codom(η)(r) .<= 1e-4)
    end
end