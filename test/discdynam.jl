using AlgebraicDynamics.DiscDynam
using Catlab
using Catlab.CategoricalAlgebra
using Test

@testset "DiscDynam" begin
f(x) = [x[1]*x[2], x[1]+x[2]]
g(x) = [x[1]+x[2], x[2]-x[1], x[3]]

d = DynamUWD{Float64, Function}()
add_parts!(d, :Junction,  3, jvalue=[1,1,1])
add_parts!(d, :Box,       2, dynamics=[f,g])
add_parts!(d, :State,     5, system=[1,1,2,2,2], value=[1,1,1,1,3])
add_parts!(d, :Port,      4, box=[1, 1, 2, 2], junction=[1, 2, 2, 3], state=[1,2,3,4])
add_parts!(d, :OuterPort, 2, outer_junction=[1,3])



@test isconsistent(d)
@test update!(d) == [1,3,3,0,3]
@test update!(d) == [3,4,4,-3,3]
@test isconsistent(d)

f(x) = [x[2], x[1]]
g(x) = [x[1]+x[2], x[2]-x[1]]
h(x) = [2*x[1], 2*x[1]*x[2]]

d = DynamUWD{Float64, Function}()
add_parts!(d, :Junction,  3, jvalue=[1,0,5])
add_parts!(d, :Box,       3, dynamics=[f,g,h])
add_parts!(d, :State,     6, system=[1,1, 2,2, 3,3], value=[1,1,1,5,3,5])
add_parts!(d, :Port,      5, box=[1, 1, 2, 2, 3], junction=[1, 1, 1, 3,3], state=[1,2,3,4,6])
add_parts!(d, :OuterPort, 0, outer_junction=Int[])

@assert isconsistent(d) "d is not consistent, check the initial condition"
@test update!(d) == [6, 6, 6, 29, 6, 29]
@assert isconsistent(d) "d is not consistent, check the initial condition"

end #testset


#=

Desired API for defining dynamical systems:
we need a syntax for expressing the structure of the problem
d_str = @relation (x, z) begin
    f(x, y)
    g(y, z)
end

# apply a functor to fill in the boxes with concrete dynamical systems

d_cont = Functor(:f=>Dynam(
        dynam=f(x) = [x[1]*x[2]...]
        states=2,
        portmap=(ports=>states)
        values=[-3.0, 1.2])
    :g=>Dynam(g(x) = states, 
    ...)    
)(d_str)
# junction values can be computed by commutativity
# junction⋅jvalue = state⋅value
# such that isconsistent(d_cont) is satisfied


d = EulerFunctor(h)(d_cont)
# just takes box.dynam => h*box.dynam

for i in 1:100
    update!(d)
end
=#
