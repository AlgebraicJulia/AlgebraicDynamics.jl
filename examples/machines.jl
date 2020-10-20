module Machine
using Catlab
using Catlab.Programs
using Catlab.Graphics
using Catlab.Theories
using Catlab.CategoricalAlgebra
using Plots

export TheoryDynamMachine, AbstractDynamMachine, DynamMachine

@present TheoryDynamMachine(FreeSchema) begin
    Box::Ob
    InPort::Ob
    OutPort::Ob
    State::Ob
    
    Value::Data
    Dynamics::Data

    parameterizes::Hom(InPort, Box)
    state::Hom(OutPort, State)
    system::Hom(State, Box)
    feeder::Hom(InPort, OutPort)
    
    value::Attr(State, Value)
    dynamics::Attr(Box, Dynamics)
end

const AbstractDynamMachine = AbstractACSetType(TheoryDynamMachine)
const DynamMachine = ACSetType(TheoryDynamMachine, index=[:parameterizes, :state, :system, :value, :feeder, :dynamics])
DynamMachine() = DynamMachine{Real, Function}()



function update!(dm::AbstractDynamMachine) 
    boxes = 1:nparts(dm, :Box)
    newvalues = map(boxes) do b
        ivalues = subpart(dm, subpart(dm, incident(dm, b, :parameterizes), :feeder), :value)
        xvalues = subpart(dm, incident(dm, b, :system), :value)
        dynamics = subpart(dm, b, :dynamics)
        return dynamics(ivalues, xvalues)
    end

    map(boxes, newvalues) do b, nv
        set_subpart!(dm, incident(dm, b, :system), :value, nv)
    end
end

# f = (i, x) -> [x[1] + 1]
# g = (i, y) -> [i[1]*y[1]]
# dm = DynamMachine()
# add_parts!(dm, :Box,    2, dynamics = [f,g])
# add_parts!(dm, :State,  2, system = [1,2], value=[2, 1])
# add_parts!(dm, :OutPort,1, state = [1])
# add_parts!(dm, :InPort, 1, parameterizes = [2], feeder = [1])


# for i=1:4 
#     update!(dm)
#     @show subpart(dm, [1,2], :value)
# end





α = 1.2
β = 0.1
γ = 1.3
δ = 0.1

dotr = (i, r) -> [α*r[1] - β*i[1]*r[1]]
dotf = (i, f) -> [-γ*f[1] + δ* i[1]* f[1]]

function eulers(f::Function, h::Float64) 
    return (i, x) -> x + h*f(i, x)
end

h = 0.1

dm = DynamMachine()
add_parts!(dm, :Box,    2, dynamics = [eulers(dotr, h), eulers(dotf, h)])
add_parts!(dm, :State,  2, system = [1,2], value=[10, 10])
add_parts!(dm, :OutPort,2, state = [1,2])
add_parts!(dm, :InPort, 2, parameterizes = [1,2], feeder = [2,1])

N = 100

ts = map(t -> t*h, 1:N)
rs = Array{Float64,1}(undef, N)
fs = Array{Float64,1}(undef, N)
for i=1:N  
    update!(dm)
    vs = subpart(dm, [1,2], :value)
    rs[i] = vs[1]
    fs[i] = vs[2]
end

plot(ts, rs)
plot!(ts, fs)
@show rs
@show fs
end #module