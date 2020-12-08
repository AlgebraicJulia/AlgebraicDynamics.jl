using AlgebraicDynamics.Machines
using Catlab.WiringDiagrams
using Test


@testset "Machines" begin
# Identity 
A = [:A]
uf(x,p) = [p[1] - x[1]]
rf(x) = x
mf = Machine{Float64}(1,1,1, uf, rf)

f = Box(:f, A, A)
m_id = oapply(singleton_diagram(f), [mf])

x0 = 1
p0 = 200
@test m_id.update([x0], [p0]) == [p0 - x0]
@test m_id.readout([x0], [p0]) == [x0]

# composite
d = WiringDiagram(A, A)
f1 = add_box!(d, f)
f2 = add_box!(d, f)
add_wires!(d, Pair[
    (input_id(d), 1) => (f1, 1),
    (f1, 1) => (f2, 1),
    (f2, 1) => (output_id(d), 1)
])
m12 = oapply(d, [mf, mf])

x0 = -1
y0 = 29
p0 = 200
@test m12.update([x0, y0], [p0]) == [p0 - x0, x0 - y0]
@test m12.readout([x0,y0], [p0]) == [y0]


# break and back together
d = WiringDiagram(A,A)
f1 = add_box!(d, f)
f2 = add_box!(d, f)
add_wires!(d, Pair[
    (input_id(d), 1) => (f1, 1),
    (input_id(d), 1) => (f2, 1),
    (f1, 1) => (output_id(d), 1),
    (f2, 1) => (output_id(d), 1)
])
m = oapply(d, [mf, mf])

x0 = -1
y0 = 29
p0 = 200
@test m.update([x0, y0], [p0]) == [p0 - x0, p0 - y0]
@test m.readout([x0,y0], [p0]) == [x0 + y0]

@test m.nparams == 1
@test m.nstates == 2
@test m.noutputs == 1
end