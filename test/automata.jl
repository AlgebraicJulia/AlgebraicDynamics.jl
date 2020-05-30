@testset "Automata" begin
# Declare that states are modeled as Int
Z = Ob(FreeSMC, Int)

# Create decorating matrices for counters mod 3 and mod 2
counter3_matrix = reshape([0, 1, 0, 0, 0, 1, 1, 0, 0], 3,3)
counter2_matrix = reshape([0, 1, 1, 0], 2, 2)

counter3 = Hom(System([i%2 for i in 1:3], [1, 2], [i%2 for i in 1:3],  counter3_matrix), Z, Z)
counter2 = Hom(System([i%2 for i in 1:2], [1, 2], [i%2 for i in 1:2], counter2_matrix), Z, Z)

traj = nsteps(counter3, 1:4)
@test length(traj) == 4
@test map(sort, traj) == [
    [1, 2],
    [2, 3],
    [1, 3],
    [1, 2]
]


align = compose(counter2, counter3)
traj = nsteps(align, 1:4)
@test length(traj) == 4
@test map(sort, traj) == [
    [(1, 1), (2, 2)],
    [(1, 3), (2, 2)],
    [(1, 3)],
    []
]

prod = otimes(counter2, counter3)
curr = initial(prod)


traj = nsteps(prod, 1:4)
@test length(traj) == 4
@test map(sort, traj) == [
    [(1, 1), (1, 2), (2, 1), (2, 2)],
    [(1, 2), (1, 3), (2, 2), (2, 3)],
    [(1, 1), (1, 3), (2, 1), (2, 3)],
    [(1, 1), (1, 2), (2, 1), (2, 2)],
]

@testset "Recursive Automata" begin
f = (counter3⋅counter3)⊗(counter2)
initial(f)
@time next(f, initial(f))
@time next(f, initial(f))
@test next(f, initial(f)) == [((2, 2), 2), ((3, 3), 2), ((2, 2), 1), ((3, 3), 1)]

f = (counter3⋅counter3)⊗(counter3⋅counter2)
@time next(f, initial(f))
@time next(f, initial(f))
@test next(f, initial(f)) == [((2, 2), (2, 2)), ((3, 3), (2, 2)), ((2, 2), (3, 1)), ((3, 3), (3, 1))]

f = (counter3⋅counter2)⊗(counter3⋅counter2)
@time next(f, initial(f))
@time next(f, initial(f))
@test next(f, initial(f)) == [((2, 2), (2, 2)), ((3, 1), (2, 2)), ((2, 2), (3, 1)), ((3, 1), (3, 1))]

f = (counter2⊗(counter3⋅counter2))⊗(counter3⋅counter2)
@time next(f, initial(f))
@time next(f, initial(f))
@test next(f, initial(f)) == [(2, (2, 2)), (1, (2, 2)), (2, (3, 1)), (1, (3, 1))]
end
end
