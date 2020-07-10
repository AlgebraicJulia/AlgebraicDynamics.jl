using Test

@test src(FG(compose(edge, edge)).decoration) == [1,2]
@test tgt(FG(compose(edge, edge)).decoration) == [2,3]

@test src(FG(otimes(edge, edge)).decoration) == [1,3]
@test tgt(FG(otimes(edge, edge)).decoration) == [2,4]

@test src(FG((edge⋅edge)⊗(edge⋅edge)).decoration) == [1,2,4,5]
@test tgt(FG((edge⋅edge)⊗(edge⋅edge)).decoration) == [2,3,5,6]
