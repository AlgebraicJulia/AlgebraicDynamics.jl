module Laplacians
using Catlab
using Catlab.Theories
using Catlab.Programs
using Catlab.CategoricalAlgebra.ShapeDiagrams
using Catlab.CategoricalAlgebra.FinSets
using Catlab.CategoricalAlgebra.Graphs
using Catlab.WiringDiagrams
using Catlab.Graphics
using AutoHashEquals

import Catlab.Theories: dom, codom, id, compose, ⋅, ∘, otimes, ⊗, munit,
                        braid, σ, mcopy, Δ, mmerge, ∇, create, □, delete, ◊,
                        pair, copair, proj1, proj2, coproj1, coproj2

export Meshes, FG, GraphFunctor

display_wd(ex) = to_graphviz(ex, orientation=LeftToRight, labels=true);
id(args...) = foldl((x,y)->id(x) ⊗ id(y), args);


path(n::Int) = begin
    g = Graph()
    add_vertices!(g, n)
    add_edges!(g, 1:(n-1),2:n)
    return g
end
cycle(n::Int) = begin
    g = Graph()
    add_vertices!(g, n)
    add_edges!(g, 1:(n-1),2:n)
    add_edge!(g, n, 1)
    return g
end


k₂ = path(2)
k₃ = cycle(3)
c₄ = cycle(4)

@auto_hash_equals struct Vertices
    n::Int
end
Base.eachindex(V::Vertices) = 1:V.n

struct GraphDecorator <: AbstractFunctor end
struct GraphLaxator <: AbstractLaxator end
const GraphFunctor = LaxMonoidalFunctor{GraphDecorator, GraphLaxator}
id(::Type{GraphFunctor}) = GraphFunctor(GraphDecorator(), GraphLaxator())

function (pd::GraphDecorator)(n::FinOrd)
    return g -> typeof(g) <: Graph && nv(g) == n.n
end

function (pd::GraphDecorator)(f::FinOrdFunction)
    Ff(g) = begin
        g′ = Graph()
        add_vertices!(g′, codom(f).n)
        add_edges!(g′, f.(src(g)), f.(tgt(g)))
        return g′
    end
    return Ff
end

function (l::GraphLaxator)(g::Graph, h::Graph)
    g′ = deepcopy(g)
    add_vertices!(g′, nv(h))
    add_edges!(g′, src(h) .+ nv(g), tgt(h) .+ nv(g))
    return g′
    # return Graph(collect(1:nv(g)+nv(h))),
    #                    vcat(edges(g), map(x->x+length(nv(g)), edges(h))))
end


const GraphCospan = DecoratedCospan{GraphFunctor, Graph}


function (::Type{GraphCospan})(l::AbstractVector, g::Graph, r::AbstractVector)
    return GraphCospan(Cospan(FinOrdFunction(l, nv(g)),
                              FinOrdFunction(r, nv(g))),
                       id(GraphFunctor), g)
end

function otimes(f::FinOrdFunction, g::FinOrdFunction)
    n′ = FinOrd(dom(f).n + dom(g).n)
    m′ = FinOrd(codom(f).n + codom(g).n)
    f′(X) = x <= dom(f).n ? f(x) : g(x-dom(f).n)+codom(f).n
    FinOrdFunction(f′, n′, m′)
end

@instance BiproductCategory(Vertices, GraphCospan) begin
    dom(f::GraphCospan) = Vertices(dom(left(f)).n)
    codom(f::GraphCospan) = Vertices(dom(right(f)).n)

    compose(p::GraphCospan, q::GraphCospan) = begin
        # reimplementation of pushout of Span{FinOrdFunc, FinOrdFun}
        # to save the value of coeq
        f, g = right(p), left(q)
        coprod = coproduct(codom(f), codom(g))
        ι1, ι2 = left(coprod), right(coprod)
        coeq = coequalizer(f⋅ι1, g⋅ι2)
        f′, g′ = ι1⋅coeq, ι2⋅coeq
        composite = Cospan(left(p)⋅f′, right(q)⋅g′)
        dpuq = decorator(p).L(decoration(p), decoration(q))
        return GraphCospan(composite, decorator(p), decorator(p).F(coeq)(dpuq))
    end

    id(X::Vertices) = GraphCospan(
        Cospan(id(FinOrd(X.n)), id(FinOrd(X.n))),
        id(GraphFunctor),
        EmptyGraph(X.n))

    otimes(X::Vertices, Y::Vertices) = Vertices(X.n + Y.n)

    otimes(f::GraphCospan, g::GraphCospan) = begin
        fl, fr = left(f), right(f)
        gl, gr = left(g), right(g)
        f′, g′ = otimes(fl, gl), otimes(fr, gr)
        G = decorator(f).L(decoration(f), decoration(g))
        return GraphCospan(Cospan(f′, g′), decorator(f), G)
    end

    munit(::Type{Vertices}) = Vertices(0)

    braid(X::Vertices, Y::Vertices) = begin
        Z = otimes(X, Y)
        GraphCospan(
            Cospan(
                id(FinOrd(Z.n)),
                FinOrdFunction(vcat(X.n+1:Z.n, 1:X.n), Z.n, Z.n)
            ), id(GraphFunctor), EmptyGraph(Z.n))
    end

    mcopy(X::Vertices) = GraphCospan(
        Cospan(
            id(FinOrd(X.n)),
            FinOrdFunction(vcat(1:X.n,1:X.n), 2*X.n, X.n)
        ), id(GraphFunctor), EmptyGraph(X.n))

    mmerge(X::Vertices) = GraphCospan(
        Cospan(
            FinOrdFunction(vcat(1:X.n,1:X.n), 2*X.n, X.n),
            id(FinOrd(X.n))
        ), id(GraphFunctor), EmptyGraph(X.n))

    create(X::Vertices) = GraphCospan(
        Cospan(FinOrdFunction(Int[], 0, X.n), id(FinOrd(X.n))),
        id(GraphFunctor), EmptyGraph(X.n))

    delete(X::Vertices) = GraphCospan(
        Cospan(id(FinOrd(X.n)), FinOrdFunction(Int[], 0, X.n)),
        id(GraphFunctor), EmptyGraph(X.n))

    pair(f::GraphCospan, g::GraphCospan) = compose(mcopy(dom(f)), otimes(f, g))
    copair(f::GraphCospan, g::GraphCospan) = compose(otimes(f, g), mmerge(codom(f)))

    proj1(A::Vertices, B::Vertices) = otimes(id(A), delete(B))
    proj2(A::Vertices, B::Vertices) = otimes(delete(A), id(B))

    coproj1(A::Vertices, B::Vertices) = otimes(id(A), create(B))
    coproj2(A::Vertices, B::Vertices) = otimes(create(A), id(B))
end


@present Meshes(FreeBiproductCategory) begin
    V::Ob
    edge::Hom(V,V)
    triangle::Hom(V⊗V, V)
    square::Hom(V⊗V, V⊗V)
end

V, edge, triangle, square = generators(Meshes);

FunctorGenerators = Dict(
    edge => GraphCospan([1], k₂, [2]),
    triangle => GraphCospan([1,2], k₃, [3]),
    square => GraphCospan([1,2], c₄, [3,4]),
    V => Vertices(1),
)
FG(ex) = functor((Vertices, GraphCospan), ex, generators=FunctorGenerators)
end
