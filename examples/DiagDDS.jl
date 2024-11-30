# # Creating Hybrid Systems
# This examples explores the idea of creating hybrid dynamical systems with diagrams.
# The first step is to do the imports.

using Catlab
using Catlab.Theories
using Catlab.Present
using Catlab.CategoricalAlgebra
using LinearAlgebra
using Catlab.Graphs
using Catlab.Graphics
using Catlab.Programs.DiagrammaticPrograms
using Catlab.Programs.DiagrammaticPrograms: NamedGraph

function Graphs.BasicGraphs.Graph(R::AbstractMatrix)
  n = size(R,1)
  G = BasicGraphs.Graph(n)
  for i in 1:n
    for j in 1:n
      if j == i
        continue
      end
      if R[i,j] != 0
        add_edge!(G, i, j)
      end
    end
  end
  return G
end


# We want to use discrete dynamical systems as our model of dynamics because they are the simplest form of dynamical system worthy of the name.
# We have a state space, and a process `next` that sends each state to the next state.

@present SchDDS(FreeSchema) begin
  X::Ob
  next::Hom(X,X)
end

# Create the concrete julia type of a DDS and define a DDS homomorphism based on the schema above.

@acset_type DDS(SchDDS, index=[:next])

# Here we create a DDS on 3 states that orbits in a cycle. This is the connoncial discrete cycle of length 3.
# 1 ↦ 2 ↦ 3 ↦ 1

X₀ = @acset DDS begin
  X = 3
  next = [2,3,1]
end

# Now we create a system that has a 3 cycle embedded in it (as the first 3 states) and then another part that sends 5 ↦ 6 ↦ 1
# This system also has only one orbit. After a brief transient (if you start in states 5 or 6) the system becomes 1 ↦ 2 ↦ 3 ↦ 1.

X₁ = @acset DDS begin
  X = 6
  next = [2,3,1, 5, 6, 1]
end

# Now we create another system that has a 3 cycle embedded in it (as the first 3 states) and then another part that is a flipflop 4 ↦ 5 ↦ 5.
# This system has 2 orbits. The behavior is either 1 ↦ 2 ↦ 3 ↦ 1 or 4 ↦ 5 ↦ 4 depending on the initial state. 

X₂ = @acset DDS begin
  X = 5
  next = [2,3,1, 5, 4]
end

# We can check that the X₀ system embeds into both other systems. By checking the naturality of the relevent ACSetTransformation.

f = ACSetTransformation(X₀, X₁, X=[1,2,3])
@assert is_natural(f)
g = ACSetTransformation(X₀, X₂, X=[1,2,3])
@assert is_natural(g)

# Since we have created a span in DDS, we can construct the pushout with the colimit function. 
# This will create a new system that is the colimit of both substems. 
# Catlab will comput not just a colimit object but the entire colimiting cocone.

Ycocone = colimit(Span(f,g));

# The apex of the cocone is what you usually call the colimit object. It has 6 + 5 - 3 = 8 states.

Y = apex(Ycocone)

# I had thought that diagrams in spaces of dynamical systems would be the key formulating hybrid dynamical systems, but this example makes it clear that you don't want to do that.
# In setting up the diagram in DDS, you see that the image of each each in the diagram must be closed under the dynamics.
# If it wasn't closed, then evolving in the image would take you out of the image, which contradicts the fact that it is the image of a DDS.

f′ = ACSetTransformation(X₀, X₁, X=[2,3,4])

@assert !is_natural(f′)

# So a hybrid dynamical system can't be a diagram with respect to the dynamics. 
# It has to a diagram with respect to the spaces and then a dynamics on each object in the diagram.
# We are rewarded for looking at DDS instead of continuous dynamical systems because the state space of a DDS is just a finite set.
# So a diagram of state spaces will just be a diagram in FinSet, which Catlab has really good support for.

S₀ = FinSet(3)
S₁ = FinSet(6)
S₂ = FinSet(5)

f = FinFunction([1,2,3], S₀, S₁)
g = FinFunction([1,2,3], S₀, S₂)
diagram = FreeDiagram([S₀,S₁,S₂], [(f,1,2),(g,1,3)])

# Now that we have related the state spaces, we can put in dynamics that are more flexible. 

X₀ = X₀

# X₁ is the 6-cycle

X₁ = @acset DDS begin
  X = 6
  next = [2, 3, 4, 5, 6, 1]
end

# X₂ will progress from 1 ↦ 2 ↦ 3 ↦ 4 and then get stuck in a flip-flop 4 ↦ 5 ↦ 4

X₂ = @acset DDS begin
  X = 5
  next = [2, 3, 4, 5, 4]
end

# We can implement a simple simulator with some logging to record the trajectory for further analysis.

"""    HybridSystem

A hybrid systems is a diagram amongst the state spaces and a dynamics for each object
"""
struct HybridSystem
  diagram::FreeDiagram
  dynamics::Vector{DDS}
end

"""    HybridState

A hybrid state knows what system it lives in.
"""
struct HybridState
  system::Int # identifier of the system you are in.
  state::Int  # state within that system
end

"""    step(system::HybridSystem, state::HybridState)

Take a step of the internal dynamics of the current system.
"""
function step(system::HybridSystem, state::HybridState)
  X = system.dynamics[state.system]
  return HybridState(state.system, X[state.state, :next])
end

"""    backjump(system::HybridSystem, state::HybridState)

Compute the set of available jumps into the transition set.
They are grouped by the system that they go to.
"""
function backjump(system::HybridSystem, state::HybridState)
  jumpedges = incident(system.diagram, state.system, :tgt)
  jumpsrcs = system.diagram[jumpedges, :src]
  fs = system.diagram[jumpedges, :hom]
  jumpsets = [(i, preimage(f, state.state)) for (i, f) in zip(jumpsrcs, fs)]
  ## if this preimage is empty we can't actually go there
  jumpsets = filter( p -> !isempty(p[2]), jumpsets)
  return jumpsets
end

"""    forwardjump(system::HybridSystem, state::HybridState)

Compute the set of available jumps out of the transition set.
"""
function forwardjump(system::HybridSystem, state::HybridState)
  jumpedges = incident(system.diagram, state.system, :src)
  jumptgts = system.diagram[jumpedges, :tgt]
  fs = system.diagram[jumpedges, :hom]
  jumpsets = [(i, f(state.state)) for (i, f) in zip(jumptgts, fs)]
  ## no need to check for empty images
  return jumpsets
end

abstract type SimulationStep end

struct DynamStep <: SimulationStep
  step::Int
  system::Int
  priorstate::HybridState
  nextstate::HybridState
end

struct BackJump <: SimulationStep
  step::Int
  priorsystem::Int
  priorstate::HybridState
  nextstate::HybridState
  options::Vector{Tuple{Int, Vector{Int}}}
end

struct ForwardJump <: SimulationStep
  step::Int
  priorsystem::Int
  priorstate::HybridState
  nextstate::HybridState
  options::Vector{Tuple{Int, Int}}
end

struct SimulationTrace
  states::Vector{HybridState}
  steps::Vector{SimulationStep}
end

"""    step!(hds::HybridSystem, log::SimulationTrace, i::Int)

Take a single internal step of the system.
"""
function step!(hds::HybridSystem, log::SimulationTrace, i::Int)
    states = log.states
    steps = log.steps
    priorstate = states[end]
    state = step(hds, states[end])
    push!(states, state)
    push!(steps, DynamStep(i, state.system, priorstate, state))
    return state
end

abstract type JumpStyle end
struct JumpFirst <: JumpStyle end
struct JumpRandom <: JumpStyle end

"""    jump!(hds::HybridSystem, log::SimulationTrace, i::Int, jumpsets, type::JumpStyle)

Try to perform a jump. The JumpStyle argument controls the behavior for how to jump.

1. JumpFirst implies that you will deterministically take the first available jump.
2. JumpRandom takes a randomly available jump.

A Jump step is composed of a backjump, an internal jump in the jump system, then a forward jump out of that jump system.
"""
function jump!(hds::HybridSystem, log::SimulationTrace, i::Int, jumpsets, type::JumpStyle)
  ## println("Reverse Jump\t $jumpsets")
  ## when we go into a jump we have to record where we came from.
  states = log.states
  steps = log.steps
  state = states[end]
  fromsystem = state.system
  priorstate = state
  if type == JumpFirst()
    state = HybridState(jumpsets[1][1], jumpsets[1][2][1])
    push!(states, state)
    push!(steps, BackJump(i, fromsystem, priorstate, state, jumpsets))
  end
  if type == JumpRandom()
    jumpto = rand(1:length(jumpsets))
    options = jumpsets[jumpto][2]
    toidx = rand(1:length(options))
    state = HybridState(jumpsets[jumpto][1], jumpsets[jumpto][2][toidx])
    push!(states, state)
    push!(steps, BackJump(i, fromsystem, priorstate, state, jumpsets))
  end
  priorstate = state
  state = step(hds, state)
  println("Jump Step\t $state")
  push!(states, state)
  push!(steps, DynamStep(i, state.system, priorstate, state))
  jumpset = forwardjump(hds, state)
  ## don't want to go back where we came from
  jumpset = filter(p-> p[1] != fromsystem, jumpset)
  println("Forward Jump\t $jumpset")
  priorstate = state
  if type == JumpFirst()
    state = HybridState(jumpset[1]...)
  end
  if type == JumpRandom()
    toidx = rand(1:length(jumpset))
    state = HybridState(jumpset[toidx]...)
  end
  push!(states, state)
  push!(steps, ForwardJump(i, fromsystem, priorstate, state, jumpset))
  return state
end

"""    simulate(hds::HybridSystem, state₀::HybridState, nsteps)

1. Take a step of internal dynamics
2. Try to jump

See step! and jump! for details.
"""
function simulate(hds::HybridSystem, state₀::HybridState, nsteps::Int, type::JumpStyle)
  states = [state₀]
  steps = SimulationStep[]
  log = SimulationTrace(states, steps)
  for i in 1:nsteps
    state = step!(hds, log, i)
    println("Dynam Step\t $state")
    jumpsets = backjump(hds, state)
    if !isempty(jumpsets)
      if type == JumpRandom() && rand(Bool)
        jump!(hds, log, i, jumpsets, type)
      end
    end
  end
  return states, steps
end

# Now that we have a simulator, we can run some simulations.

hds = HybridSystem(diagram, [X₀, X₁, X₂])
states, steps = simulate(hds, HybridState(2, 1), 10, JumpFirst());

# With Random Jumps, this will be different every time.
states, steps = simulate(hds, HybridState(2, 1), 10, JumpRandom());

# Since the system is now nondeterministic, we need to switch from a simple simulation, to an analyis of reachability that looks at all possible jumps. 
# For that, we need to start computing adjacency matrices and reachability relations.
# The `adjacency_matrix` of a single DDS is easy to compute.

function adjacency_matrix(dds::DDS)
  n = nparts(dds, :X)
  A = zeros(Bool, n, n)
  for i in 1:n
    j = dds[i,:next]
    A[j,i] = 1 
  end
  return A
end

@assert adjacency_matrix(X₀) * [1;0;0] == [0;1;0]
@assert adjacency_matrix(X₀) * [0;1;0] == [0;0;1]
@assert adjacency_matrix(X₀) * [0;0;1] == [1;0;0]
@assert adjacency_matrix(X₀)^3 == I

# For the HDS, we need to incorporate the same jumping rules from our simulator into the process for computing the adjacency matrix.

function adjacency_matrix(hds::HybridSystem)
  composite = coproduct(hds.dynamics)
  A = adjacency_matrix(apex(composite))
  for sys in 1:length(hds.dynamics)
    for state in parts(hds.dynamics[sys], :X)
      ## handle the edges that come from back jumps
      jumpsets = backjump(hds, HybridState(sys, state))
      for k in 1:length(jumpsets)
        tosys = jumpsets[k][1]
        for l in jumpsets[k][2]
          fromstate = legs(composite)[sys][:X](state)
          tostate = legs(composite)[tosys][:X](l)
          A[tostate, fromstate] = 1
        end
      end
      ## handle the edges that come from forward jumps
      jumpset = forwardjump(hds, HybridState(sys, state))
      ## can't implement this rule here because we don't know where we jumped out of. 
      ## jumpset = filter(p-> p[1] != fromsystem, jumpset)
      for k in 1:length(jumpset)
        tosys = jumpset[k][1]
        l = jumpset[k][2]
        fromstate = legs(composite)[sys][:X](state)
        tostate = legs(composite)[tosys][:X](l)
        A[tostate, fromstate] = 1
      end
    end
  end
  return A
end

adjacency_matrix(hds)

# We can also draw the adjacency matrix of any system as a graph. To see the behavior visually.

Catlab.Graphs.Graph(hds::HybridSystem) = Graph(adjacency_matrix(hds))

Catlab.Graphics.to_graphviz(hds::HybridSystem, args...; kwargs...) = to_graphviz(Graph(hds), args...;kwargs...)

to_graphviz(hds, node_labels=true)


# The reachability relation can be compute using the power method.

"""    reachability_matrix(hds::HybridSystem, maxiter::Int)

We can compute the reachability relation of the hybrid dynamical system by the power method.
This works on the same principle as the breadth first search in the language of linear algebra.
Each multiplication by the adjacency matrix expands the number of steps by one. We are computing
the matrix whose rows and columns are states and R[i,j] is 1 iff there exists a trajectory that
starts at j and ends at i.
"""
function reachability_matrix(hds::Union{DDS, HybridSystem}, maxiter::Int)
  A = adjacency_matrix(hds)
  R = I(size(A)[1])
  for i in 1:maxiter
    R′ = R .| (A*R .!= 0)
    if R == R′
      return R, i
    end
    R = R′
  end
  return R, maxiter
end

# First we look at the reachability relation for the coprpduct system. 
# This shows us what happens if each system evolves through time without interacting
# via the jumps.

R₀, iₘ₀ = reachability_matrix(coproduct(hds.dynamics)|>apex, 100)
R₀


# Now we look at the reachability relation for the hybrid system. 
# This shows us what happens if each system evolves through time with interacting
# via the jumps. We aren't keeping track of the length of the trajectory so we can't use this method for orbits.

R, iₘ = reachability_matrix(hds, 15)
R

# Now we can build some more complex systems and look at their reachability.

# C₄ is the four cycle.

C₄ = @acset DDS begin
  X = 4
  next = [2,3,4,1]
end

# D₂ is the discrete system on two states. The only operation is to stay still.

D₂ = @acset DDS begin
  X = 2
  next = [1, 2]
end

# Let's build a bigger diagram.

S₂ = FinSet(2)
S₄ = FinSet(4)
f  = FinFunction([1,4], S₂, S₄)
diagram = FreeDiagram([S₄, S₂, S₄, S₂, S₄],
    [(f,2,1),(f,2,3),(f,4,3), (f,4,5)])

hds = HybridSystem(diagram, [C₄, D₂, C₄, D₂, C₄])
to_graphviz(hds, node_labels=true)

# And we can run the simulation

states, steps = simulate(hds, HybridState(1, 1), 10, JumpRandom());

# This system has the following reachability matrix.

R, k = reachability_matrix(hds, 100);
println("Number of steps to converge = $k")
R

# In order to get some more nontrivial reachability, let's introduce some dead ends to our dynamics.

P₄ = @acset DDS begin
  X = 4
  next = [2,3,4,4]
end

hds = HybridSystem(diagram, [P₄, D₂, P₄, D₂, P₄])
states, steps = simulate(hds, HybridState(1, 1), 10, JumpRandom());
steps

# And our reachability matrix is much more sparse.

R, k = reachability_matrix(hds, 100);
R

# Another way to introduce complex behavior is to add some flip-flops.

F₄ = @acset DDS begin
  X = 4
  next = [3,4,1,2]
end

hds = HybridSystem(diagram, [F₄, D₂, P₄, D₂, F₄])
to_graphviz(hds, node_labels=true)

# And we can run the simulation

states, steps = simulate(hds, HybridState(1, 1), 10, JumpRandom());
steps

R, k = reachability_matrix(hds, 100);
R'


# Notice that a diagram of deterministic dynamical systems becomes 
# a nondeterministic dynamical system when we add the jumps.
# However, all the nondeterminism comes from the jumps.
# If we have constraints like every state has at most one jump point and
# a rule for allocating steps between internal steps and jumps,
# then we would still have a deterministic system. 
# These properties can be statically checked from the diagram between the spaces.
# 
# So far we have just used the compositional structure for specifying the system.
# An open research question is how the compositional structure can be used for analysis.
# In this case, reachability analysis can be conducted hierarchically. 
# In order for a hybrid state (i,j) to get to state (k,l) there must be path from i to k
# in the diagram of states, and that path must be consistent with j and l as the start and end
# of that path.

# Another research question is how the category of diagrams structure would help you
# understand the relationships between hybrid systems. For example, morphism of diagrams
# induce a morphism of of dynamical systems by precomposition. 
# We can characterize relationships between HDSes with properties of these morphisms.
# For example, a monic functor with identity components would give you a restriction of
# systems to certain discrete modes.

# We can make our diagram of state spaces

diagram = FreeDiagram([S₄, S₂, S₄, S₂, S₄],
    [(f,2,1),(f,2,3),(f,4,3), (f,4,5)])
D =  FinDomFunctor(diagram)

# Then formulate the shape of our subdiagram

J = FinCat(@graph begin
  x; a; y
  a → x
  a → y
end)

# We relate the shape of our subdiagram to the shape of the diagram.
# In this case, we are picking out the first three vertices and first two edges.
# Composing functors, we can compute the subdiagram, which has the information we need to specify a subsystem of D.

F = FinFunctor((V=[1,2,3], E=[[1],[2]]), J, dom(D));
FD = compose(F, D);
FreeDiagram(FD)

# Whenever you define something by precomposition, you get a restriction map. 
# In this case we get a restriction of HDS. By selecting the first three modes of the system.

simulate(HybridSystem(FreeDiagram(FD), [F₄, D₂, P₄]), HybridState(1,1), 10, JumpRandom());

# We can use categorical logic to articulate relationships between systems based
# on these functors. For example, taking the monic functors induced by subgraphs will
# give us a biheyting algebra of subsystems based on the lattice of subgraphs of a fixed
# graph. We would be able to generalize this beyond the analysis of subgraphs by looking at
# functors that are not induced by subgraphs. Based on our previous experience,
# that generalization should lead to a notion of refinement that lets you think about mult-scale phenomena.