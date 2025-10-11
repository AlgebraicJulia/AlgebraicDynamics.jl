using AlgebraicDynamics.ThresholdLinear
using Catlab

using Base.Iterators: product
using Combinatorics: powerset
using Test

@testset "Supports" begin

    @test Support() == Support(Int[])
    @test Support(1,2,3.0) == Support(1,2,3)
    @test Support(1,0,4.0) == Support(1,3)
    @test Support(1,1e-15,1e-11) == Support(1,3)
    @test Support([1e-12, 1e-11]) == Support([2])
    @test Support([1e-13]) == Support()

    # sorting
    s = Support(2, 1, 4)
    sort!(s)
    @test s == Support(1, 2, 4)

    # shifting
    s = Support(1, 2, 3.0)
    @test shift(s, 2) == Support(3, 4, 5)

    a = Support(1, 2, 3)
    b = Support(4, 5, 6)
    c = Support(1, 3, 4)

    fp1 = FPSections(a)
    fp2 = FPSections([b, c])

    @test collect(Base.product([fp1, fp2])) == [Support(1,2,3,4,5,6), Support(1,2,3,4)]

    @test ThresholdLinear.disjoint_union(fp1, fp2) == FPSections([Support([1,2,3]), Support([4,5,6]), Support([1,3,4]), Support([1,2,3,4,5,6]), Support([1,2,3,4])])

end

@testset "FPSections" begin

    c3 = CycleGraph(3)
    k4 = CompleteGraph(4)
    d5 = DiscreteGraph(5)

    @test FPSections(c3) == FPSections(Support[Support(1,2,3)])
    @test FPSections(k4) == FPSections(Support[Support(1:4)])
    @test_broken FPSections(d5) == FPSections([Support(k) for k in powerset(1:5)])

end
