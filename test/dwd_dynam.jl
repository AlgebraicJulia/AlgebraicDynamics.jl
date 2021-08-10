using Test
using LabelledArrays

using AlgebraicDynamics.DWDDynam
using Catlab.WiringDiagrams

# Identity 
A = [:A]
uf(u, x, p, t) = [x[1] - u[1]]
rf(u, args...) = u
mf = ContinuousMachine{Float64}(1,1,1, uf, rf)

f = Box(:f, A, A)
d_id = singleton_diagram(f)
m_id = oapply(d_id, [mf])

x0 = 1
p0 = 200
@test eval_dynamics(m_id, [x0], [p0]) == [p0 - x0]
@test readout(m_id, [x0]) == [x0]

# unfed parameter 
d12 = WiringDiagram(A, A)
f1 = add_box!(d12, f)
f2 = add_box!(d12, f)
add_wires!(d12, Pair[
    (f1, 1) => (f2, 1),
    (f2, 1) => (output_id(d12), 1)
])
m12 = oapply(d12, mf)


# composite
add_wire!(d12, (input_id(d12), 1) => (f1, 1))
m12 = oapply(d12, [mf, mf])

x0 = -1
y0 = 29
p0 = 200
@test eval_dynamics(m12, [x0, y0], [p0]) == [p0 - x0, x0 - y0]
@test readout(m12, [x0,y0]) == [y0]


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
m = oapply(d, Dict(:f => mf))

x0 = -1
y0 = 29
p0 = 200
@test eval_dynamics(m, [x0, y0], [p0]) == [p0 - x0, p0 - y0]
@test readout(m, [x0,y0]) == [x0 + y0]

@test m.ninputs == 1
@test m.nstates == 2
@test m.noutputs == 1

# trace
d_trace = WiringDiagram(A, A)
f1 = add_box!(d_trace, f)
add_wires!(d_trace, Pair[
    (input_id(d_trace), 1) => (f1, 1),
    (f1, 1) => (f1, 1),
    (f1, 1) => (output_id(d_trace), 1)
])
m_trace = oapply(d_trace, mf)
@test nstates(m_trace) == 1
@test eval_dynamics(m_trace, [x0], [p0]) == [p0]
@test readout(m_trace, [x0]) == [x0]


# oapply and ocompose
d_tot = ocompose(d, [d12, d_id])
m_tot1 = oapply(d_tot, mf) 
m_tot2 = oapply(d, [m12, m_id])
x0 = [-1, 5.5, 20]
@test nstates(m_tot1) == 3
@test nstates(m_tot2) == 3

@test eval_dynamics(m_tot1, x0, [p0]) == eval_dynamics(m_tot2, x0, [p0])
@test eval_dynamics(m_tot1, x0, [p0]) == [p0 - x0[1], x0[1] - x0[2], p0 - x0[3]]

@test readout(m_tot1, x0) == [x0[2] + x0[3]] 
@test readout(m_tot2, x0) == [x0[2] + x0[3]] 

# big diagram
d = WiringDiagram([:A, :A], [:A, :A, :A])
b1 = add_box!(d, Box(:f, [:A, :A], [:A]))
b2 = add_box!(d, Box(:g, [:A], [:A, :A]))
b3 = add_box!(d, Box(:h, [:A], [:A]))

bin = input_id(d); bout = output_id(d)

add_wires!(d, Pair[
    (bin, 1) => (b1, 2),
    (bin, 1) => (b2, 1),
    (bin, 2) => (b3, 1),
    (b1, 1) => (bout, 1), 
    (b2, 1) => (bout, 1),
    (b2, 1) => (bout, 2),
    (b2, 2) => (b2, 1),
    (b3, 1) => (b2, 1),
    (b3, 1) => (bout, 3)
])

m1 = ContinuousMachine{Float64}(2,1,1, 
        (u, x, p, t) -> [x[1] * x[2]  - u[1]], 
        (u,p,t) -> 2*u)
m2 = ContinuousMachine{Float64}(1, 2, 2, 
        (u, x, p, t) -> [u[1]*u[2], x[1]^2*u[2]], 
        (u,p,t) -> u)

m3 = ContinuousMachine{Float64}(1,2,1, 
        (u, x, p, t) -> [u[1]^2 - x[1], u[2] - u[1]], 
        (u,p,t) -> [u[1] + u[2]])

xs = Dict(:f => m1, :g => m2, :h => m3, :j => mf)
m = oapply(d, xs)
@test ninputs(m) == 2
@test nstates(m) == 5
@test noutputs(m) == 3
u = [2.0, 3.0, 5.0, -7.0, -0.5]
x = [0.1, 11.0]
@test eval_dynamics(m, u, x) ≈ 
    [-u[1], u[2]*u[3], (x[1]+u[3]+u[4]+u[5])^2*u[3], u[4]^2 - x[2], u[5] - u[4]]
@test readout(m, u) == [2*u[1] + u[2], u[2], u[4] + u[5]]
@test readout(oapply(d_id, m3), [u[4], u[5]]) == [u[4] + u[5]]

# labeled state vectors 
m1 = ContinuousMachine{Float64}(2,1,1, 
        (u, x, p, t) -> [x[1] * x[2]  - u.a], 
        (u,p,t) -> 2*u)
m2 = ContinuousMachine{Float64}(1, 2, 2, 
        (u, x, p, t) -> [u.a*u.b, x[1]^2*u.b], 
        (u,p,t) -> u)

m3 = ContinuousMachine{Float64}(1,2,1, 
        (u, x, p, t) -> [u.a^2 - x[1], u.b - u.a], 
        (u,p,t) -> [u.a + u.b])
xs = Dict(:f => m1, :g => m2, :h => m3, :j => mf)
m = oapply(d, xs)
labeled_u = [LVector(a =2.0), LVector(a =3.0, b=5.0), LVector(a=-7.0, b=-0.5)]
@test eval_dynamics(m, labeled_u, x) ≈ 
    [-u[1], u[2]*u[3], (x[1]+u[3]+u[4]+u[5])^2*u[3], u[4]^2 - x[2], u[5] - u[4]]
@test readout(m, labeled_u) == [2*u[1] + u[2], u[2], u[4] + u[5]]

dinner = 

# eulers
h = 0.15
euler_m = oapply(d, euler_approx([m1, m2, m3], h))
@test eval_dynamics(euler_m, u, x) == u + h*eval_dynamics(m, u, x)
@test eval_dynamics(euler_m, u, x) == eval_dynamics(euler_approx(m, h), u, x)
euler_m2 = oapply(d, euler_approx([m1, m2, m3]))
@test eval_dynamics(euler_m, u, x) == eval_dynamics(euler_m2, u, x, [h])

h = 0.02
xs = Dict(:h => m3, :f => m1, :g => m2)
euler_m = oapply(d, euler_approx(xs, h))
@test eval_dynamics(euler_m, u, x) == u + h*eval_dynamics(m, u, x)
@test eval_dynamics(euler_m, u, x) == eval_dynamics(euler_approx(m, h), u, x)
euler_m2 = oapply(d, euler_approx(xs))
@test eval_dynamics(euler_m, u, x) == eval_dynamics(euler_m2, u, x, [h])
