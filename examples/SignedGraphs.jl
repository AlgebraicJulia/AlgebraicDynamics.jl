using Catlab
using Catlab.Theories
using Catlab.Programs
using Catlab.CategoricalAlgebra
using LinearAlgebra
using OrdinaryDiffEq
using Plots
using Catlab.Graphs
using Catlab.Graphics
using Catlab.Graphics.Graphviz
using Colors
using Catlab.CategoricalAlgebra.StructuredCospans
using Catlab.Programs
using Catlab.WiringDiagrams.WiringDiagramAlgebras

"""    ThSignedGraph a database schema for storing signed graphs

The parts are Vertices, Positive edges, Negative edges, and Attributes.
The subparts are psrc, ptgt, nsrc, ntgt for the positive and negative src and target functions, 
with pfunc, nfunc storing the edge function for either positive or negative edges.
"""
@present ThSignedGraph(FreeSchema) begin
    (V,P,N)::Ob
    A::AttrType
    # the positive graph
    psrc::Hom(P, V)
    ptgt::Hom(P, V)
    # the negative graph
    nsrc::Hom(N, V)
    ntgt::Hom(N, V)
    # the edge attributes
    pfunc::Attr(P,A)
    nfunc::Attr(N,A)
end

# generate the data structures, getters/setters, constructors, etc. 
@abstract_acset_type AbstractSignedGraph
@acset_type SignedGraph(ThSignedGraph, index=[:psrc,:ptgt, :nsrc, :ntgt]) <: AbstractSignedGraph

# OpenSG will be our signed graph with an interface where we can glue models together along the interfaces.
const SGOb, OpenSG = OpenACSetTypes(SignedGraph, :V)

# Tell Catlab that positive edges are green and negative edges are red.
function propertygraph(sg::SignedGraph)
    V = nparts(sg, :V)
    pg = PropertyGraph{String}(Graphs.Graph(V))
    srcs = sg[:psrc]
    tgts =  sg[:ptgt]
    add_edges!(pg, srcs, tgts, [Dict(:color => "green")])
    srcs = sg[:nsrc]
    tgts = sg[:ntgt]
    add_edges!(pg, srcs, tgts, [Dict(:color => "red")])
    return pg
end

# type alias
SG = SignedGraph{Function}

# the `@acset` macro is a DSL for constructing databases by listing out the data. 
sg = @acset SG begin
    V=3     # 3 vertices
    P = 2   # 2 positive edges
    N = 1   # 1 negative edge

    # the edgelist for P
    psrc = [1,2]
    ptgt = [2,3]

    # the edgelist for N
    nsrc = [3]
    ntgt = [1]

    # the attributes are julia functions that compute the rates
    pfunc = [sqrt, sqrt]
    nfunc = [identity]
end

"""    jac(sg::SignedGraph, u)

construct the matrix of growth rates at the point u.

TODO: pick a better name for this matrix.
"""
jac(sg::SignedGraph, u) = begin
    V = nparts(sg, :V)
    M = zeros(Float64, V,V)
    for p in parts(sg, :P)
        i = sg[p, :psrc]
        j = sg[p, :ptgt]
        α = sg[p, :pfunc]
        M[j,i] = α(u[i])
    end
    for n in parts(sg, :N)
        i = sg[n, :nsrc]
        j = sg[n, :ntgt]
        β = sg[n, :nfunc]
        M[j,i] = -β(u[i])
    end
    return M
end

"""    lotkavolterra(sg::SignedGraph)

Generate a function that computes the right hand side of the ODE for the assumption of generalized lotkavolterra semantics.
calls the `jac` function to get the matrix A in the GLV equations. You pass this function to `ODEProblem` to construct the ODE.
The intrinsic growth/decay rates are passed to the returned function as the parameters.
"""
function lotkavolterra(sg::SignedGraph)
    dudt(u,p,t) = begin
        # p is the growth rates
        # A is the community matrix
        A = jac(sg, u)
        fᵤ = p .+ A*u
        u̇ = u .* fᵤ
    end
    return dudt
end

# At this point we have defined everything we need to start simulating some systems.

# Step 1: Specify the signed graph. The functional form of the rates get captured.

sglv = @acset SG begin
    V=3
    P = 2
    N = 1
    psrc = [1,1]
    ptgt = [2,3]
    nsrc = [2]
    ntgt = [3]
    pfunc = [x->2, x->1x]
    nfunc = [x->x^2]
end

# draw the system as a signed graph
pg = propertygraph(sglv)
to_graphviz(pg)

# Step 2: set up the ODE problem.

dudt = lotkavolterra(sglv)  # function for the RHS
u₀ = [10,1,1]               # initial conditions
r = [20.0, 3.0, 3.0]        # intrinsic growth rates
dudt(u₀, -r, 0.0)           # test call to make sure it works.

# formulate and solve the modeling problem.
prob = ODEProblem(dudt, u₀, (0.0,0.5), -r)
soln = solve(prob, alg=Tsit5(), abstol=1e-4, reltol=1e-4)
p = plot(soln, vars=1:3)

# That was easy enough to solve a simple system. 
# Typing in the signed graph and the rate functions could be done in any tool.
# You probably didn't need ACT to build what we have seen so far.
# Catlab uses CT to have generic database system embedded in Julia, but you could probably do that by hand too.
# Now we are going to hierarchically build a complex system from simple components with colimits in the category of signed graphs.
# In order to define these operations wisely and derive algorithms that implement them correctly, you do need CT.
# The schema for our definition of signed graph tells us that the positive and negative edges are conceptually distinct 
# and that a map between SGs should preserve both the connectivity structure and the sign structure. 
# We use this knowledge to derive the correct notion of gluing systems along vertices.

# We need to first introduce the operad of undirected wiring diagrams. A theorem of Spivak and Fong says 
# that any mathematical structure that behaves like a system of networks sufficiently like graphs
# (ie. is a hypergraph category), is an algebra of the operad of UWDs.
# We use this theorem to justify the choice of UWDs as a primary API for compositional modeling.

# a term in the operad is like a formula. UWDs work like conjunctive normal form queries in relational algebra.
# This term says {(x,z) | ∃y, xRy ∧ yRz}. 
# In English, all the x,z pairs such that there exists a y where x is related to y and y is related to z.
term = @relation (x,z) begin
    R(x,y)
    R(y,z)
end

# Catlab knows how to draw this as a kind of bipartite port graph. 
# This is very similar to how you would draw a hypergraph as bipartite graph between vertices and hyperedges.
to_graphviz(term)

# We can specify that the R relation is actually an open SignedGraph where x should be the first vertex and y is the third vertex
R = OpenSG{Function}(sglv, FinFunction([1],3), FinFunction([3],3))

# The `oapply` function applies the operad algebra to a term to compute the composite system.
# You should think of the term as a formula that you want to compute, and the dictionary as the environment of bound variables
# Then oapply is like "eval" that uses an interpreter to evaluate the formula with that context of variable values.
sglv² = oapply(term, Dict(:R=>R))

# The result of evaluation is a new composite model that we can draw.
to_graphviz(propertygraph(apex(sglv²)))

# or simulate
dudt = lotkavolterra(apex(sglv²))
u₀ = [10,1,1]
r = [20.0, 3.0, 3.0]

# notice that we have to give 5 initial conditions and intrinsic rate paramters.
# This is because the third variable in the first system is the same as the 
# first variable in the second system and 3 + 3 - 1 == 5
prob = ODEProblem(dudt, vcat(u₀, u₀[2:end]/2), (0.0,0.5), vcat(-r, -2r[2:end]))
soln = solve(prob, alg=Tsit5(), abstol=1e-4, reltol=1e-4)
p = plot(soln)

# a helper function to avoid so many x->2's 
constant(a) = x->a

# This system is a positive feedback loop. 1 ++> 2 ++> 3 ++> 1
feedback = @acset SG begin
    V = 3
    P = 3
    psrc = [1,2,3]
    ptgt = [2,3,1]
    pfunc = constant.([1,2,1])
end

Open(sg::SignedGraph, legs...) = OpenSG{Function}(sg, map(l->FinFunction(l, nparts(sg, :V)), legs)...)
fb = Open(feedback, [1],[3])
ffn = OpenSG{Function}(sglv, FinFunction([1],3), FinFunction([3],3))

# This term says to glue the feedforward system onto another feedforward system and then introduce a feedback term.
term = @relation (x,z) begin
    ffn(x,y)
    ffn(y,z)
    feedback(z,x)
end

to_graphviz(term, box_labels=:name, junction_labels=:variable)
m = oapply(term, Dict(:ffn=>ffn, :feedback=>fb))
to_graphviz(propertygraph(apex(m)))

# Exercise: Choose initial conditions and rates to simulate the composite model m.

# dudt = lotkavolterra(apex(m))
# u₀ = 
# r = 
# prob = ODEProblem(dudt, u₀, (0.0,0.5), -r)
# soln = solve(prob, alg=Tsit5(), abstol=1e-4, reltol=1e-4)
# p = plot(soln)

# UWDs and oapply are based on composing models with colimits, which is a form of additive model assembly. 
# When you glue models, you add the number of variables and subtract the variables in the intersection.
# It works like Inclusion-Exclusion in counting for discrete probabilities.
# When we want to compose systems by stratification, we can compute a product of graphs,
# this product will have the number of vertices and edges that is the product of the factors.
# We use this for thinking about systems like having several immune chemicals that all do the same
# activation and inhibition pattern, but then have some crosstalk because the activator of one system can activate 
# the response in another system when they are mixed.

# We have to tell catlab how to multiply the rates. In this case we just mutiply them as functions elementwise.
function mult(sg::SignedGraph{Tuple{Symbol, Symbol}}, rates)
    outgraph = SignedGraph{Function}()
    #copy_parts!(outgraph, sg, :V, :P, :N)
    add_parts!(outgraph, :V, nparts(sg, :V))
    add_parts!(outgraph, :P, nparts(sg, :P))
    add_parts!(outgraph, :N, nparts(sg, :N))
    for e in parts(sg, :P)
        vars = sg[e, :pfunc]
        mult(f::Function,g::Function) = x->(f(x)*g(x))
        outgraph[e, :psrc] = sg[e, :psrc]
        outgraph[e, :ptgt] = sg[e, :ptgt]
        outgraph[e, :pfunc] = foldl(mult, rates[v] for v in vars)
        @show outgraph
    end
    for e in parts(sg, :N)
        vars = sg[e, :nfunc]
        mult(f::Function,g::Function) = x->(f(x)*g(x))
        outgraph[e, :nsrc] = sg[e, :nsrc]
        outgraph[e, :ntgt] = sg[e, :ntgt]
        @show [rates[v] for v in vars]
        outgraph[e, :nfunc] = foldl(mult, rates[v] for v in vars)
        @show outgraph
    end
    return outgraph
end


B = @acset SignedGraph{Symbol} begin
    V=3
    P = 2
    N = 1
    psrc = [1,1]
    ptgt = [2,3]
    nsrc = [2]
    ntgt = [3]
    pfunc = [:α, :β]
    nfunc = [:γ]
end

to_graphviz(propertygraph(B))

# If there was no crosstalk, we would use a discrete diagram as the second factor.
# Multipying by D₃ will make three independent copies of the initial model and scale the rates homogenously.

D₃ = @acset SignedGraph{Symbol} begin
    V = 3
    P = 3
    N = 3
    psrc = [1, 2, 3]
    ptgt = [1, 2, 3]
    pfunc = [:r₁, :r₂, :r₃]
    nsrc = [1,2,3]
    ntgt = [1,2,3]
    nfunc = [:r₁, :r₂, :r₃]
end

D₃B = product(D₃,B) |> apex
to_graphviz(propertygraph(D₃A))
D₃B

D₃Bᵐ = mult(D₃B, Dict(:α=>constant(2), :β=>identity, :γ=>x->x^2,
              :a=>constant(1), :b=>constant(1), :c=>constant(1),
              :r₁=>constant(3), :r₂=>constant(3.01), :r₃=>constant(9.02),
    ))


dudt = lotkavolterra(D₃Bᵐ)

u₀ = [10,1,1, 10,1,1,10,1,1]
r = [20.0, 3.0, 3.0,20, 3,3,20,3,3]

prob = ODEProblem(dudt, u₀, (0.0,0.5), -r)
soln = solve(prob, alg=Tsit5(), abstol=1e-4, reltol=1e-4)
p = plot(soln, vars=[1,4,7])

# As you can tell from the plot, the 3 fold replication of the model produces the same dynamics, but with different
# parameters that control the height and width of the response curve.

# We can construct a new pattern of interaction that will encode the coupling relations
A = @acset SignedGraph{Symbol} begin
    V = 3
    P = 6
    N = 3
    psrc = [1,2,3, 1, 2, 3]
    ptgt = [2,3,1, 1, 2, 3]
    pfunc = [:a, :b, :c, :r₁, :r₂, :r₃]
    nsrc = [1,2,3]
    ntgt = [1,2,3]
    nfunc = [:r₁, :r₂, :r₃]
end
to_graphviz(propertygraph(A))

# We can now couple our system in a different way. This model will have more interesting dynamics
# because the systems will interact. Notice that in the previous model there were 3 connected components
# and now there is only 1.

BA = product(B,A) |> apex
to_graphviz(propertygraph(BA))
BA

BAᵐ = mult(BA, Dict(:α=>constant(2), :β=>identity, :γ=>x->x^2,
              :a=>constant(1), :b=>constant(1), :c=>constant(1),
              :r₁=>constant(3), :r₂=>constant(2), :r₃=>constant(1),
    ))


dudt = lotkavolterra(BAᵐ)

u₀ = [10,1,1, 10,1,1,10,1,1]
r = [20.0, 3.0, 3.0,20, 3,3,20,3,3]

prob = ODEProblem(dudt, u₀, (0.0,0.5), -r)
soln = solve(prob, alg=Tsit5(), abstol=1e-4, reltol=1e-4)
p = plot(soln)

# You can change the coefficients above to change the strength of the coupling. 
# The product operation that we used is a weakly commutative monoid on the set of
# signed graphs. It will take any pair of models and give you a new model. Satisfying
# (A×B) ≃ (B×A) and (A×B)×C ≃ A×(B×C). 
# The singleton model with 1 vertex and 1 positive edge and 1 negative edge is a unit for this monoid.
#
# This product extends to a pullback which lets you define combinations of models with less more
# constrained interactions. A pullback of two models is always a submodel of the product.