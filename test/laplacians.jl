using Test
using AlgebraicDynamics
using AlgebraicDynamics.Laplacians
using Catlab.CategoricalAlgebra.ShapeDiagrams
using Catlab.CategoricalAlgebra.FinSets
using Catlab.CategoricalAlgebra.Graphs

function connected_components(g::Graph)
    s,t = FinOrdFunction(src(g), nv(g)), FinOrdFunction(tgt(g), nv(g))
    return coequalizer(s,t)
end

connected_components(g::DecoratedCospan) = connected_components(decoration(g))

cc(expr::GATExpr) = connected_components(FG(expr))

@testset "Graphs" begin
    V, edge, triangle, square = generators(Meshes);

    @test src(FG(compose(edge, edge)).decoration) == [1,2]
    @test tgt(FG(compose(edge, edge)).decoration) == [2,3]

    @test src(FG(otimes(edge, edge)).decoration) == [1,3]
    @test tgt(FG(otimes(edge, edge)).decoration) == [2,4]

    @test src(FG((edge⋅edge)⊗(edge⋅edge)).decoration) == [1,2,4,5]
    @test tgt(FG((edge⋅edge)⊗(edge⋅edge)).decoration) == [2,3,5,6]

    g = decoration(FG(square⋅triangle))
    s,t = src(g), tgt(g)
    @test s == [1, 2, 3, 4, 4, 3, 5]
    @test t == [2, 3, 4, 1, 3, 5, 4]

    g = decoration(FG(square⋅square))
    s,t = src(g), tgt(g)
    @test s == [1, 2, 3, 4, 4, 3, 5, 6]
    @test t == [2, 3, 4, 1, 3, 5, 6, 4]

    ex = (edge⋅edge)
    g = decoration(FG(ex))
    @test connected_components(g).func == [1,1,1]
    # connected_components().func == [1,1,1]

    g = decoration(FG((square⋅square)⊗(square⋅square)))
    @test connected_components(g).func == vcat(ones(6), 2ones(6))

    @test cc(square).func == ones(4)
    FG((square⊗square)⋅otimes(edge,edge,edge,edge))
    @test all(cc((square⊗square)⋅(edge⊗square⊗edge)).func .== 1)

end
