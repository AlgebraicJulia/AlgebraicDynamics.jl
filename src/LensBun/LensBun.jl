using Catlab

constfunc(x) = 1

abstract type Space end

struct EuclideanSpace
    dim::Int
end

ℝ⁰ = EuclideanSpace(0)
ℝ¹ = EuclideanSpace(1)
ℝ² = EuclideanSpace(2)

struct Lens{S,T}
    dom::S
    cod::T
    f::Function  # dom → cod
    f♯::Function # dom × tan(cod) → tan(dom)
end

dom(l::Lens) = l.dom
codom(l::Lens) = l.cod
f(l::Lens) = l.f
f♯(l::Lens) = l.f♯

id(s::EuclideanSpace) = Lens(s, s, id, (x,tx) -> tx)

function compose(l₁::Lens, l₂::Lens)
    f₁ = f(l₁)
    f♯₁ = f♯(l₁)

    f₂ = f(l₂)
    f♯₂ = f♯(l₂)
    Lens(dom(l₁), codom(l₂), f₂ ∘ f₁, (x, tz) -> f♯₁(x, f♯₂(f₁(x), tz)))
end

# vectorfield(s::EuclideanSpace, v::Function) = Lens(s, ℝ⁰, constfunc, (x,ty) -> v(x...))

# Want to take the category of elements of the functor
# Lens(Bun)(-, ℝ⁰)

struct VectorField{S<:Space}
    X::S
    v::Function # X → TX
end

Lens(vf::VectorField) = Lens(vf.X, ℝ⁰, constfunc, (x,ty)->v(x))
apply(l::Lens, v::VectorField) = compose(l, Lens(vf))

VectorField(l::Lens) = VectorField(dom(l), x->f♯(l)(x,1))

struct DynamElt
    v::VectorField
    u::VectorField
    l::Lens
end

dom(d::DynamElt) = d.u
codom(d::DynamElt) = d.v


DynamElt(l::Lens, u::VectorField) = DynamElt(VectorField(apply(l, u)), u, l)