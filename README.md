# AlgebraicDynamics.jl

[![Documentation](https://github.com/AlgebraicJulia/AlgebraicDynamics.jl/workflows/Documentation/badge.svg)](https://algebraicjulia.github.io/AlgebraicDynamics.jl/dev/)
![Tests](https://github.com/AlgebraicJulia/AlgebraicDynamics.jl/workflows/Tests/badge.svg)


AlgebraicDynamics is a library for compositional dynamical systems. We build on [Catlab.jl](https://algebraicjulia.github.io/Catlab.jl/dev/) to provide a software interfaces for the specification and solution of dynamical systems with hierarchical structure.

The general process for composing open dynamics systems is to
1. **Pick a composition syntax:** This composition syntax may be either undirected or directed. In this library, composition is implemented for the following syntaxes:
    - undirected wiring diagrams - an undireced composition syntax
    - wiring diagrams - a directed composition syntax
    - open circular port graphs (open CPGs) - a directed composition syntax

2. **Define a composition pattern:** Implement a specific undirected wiring diagram, wiring diagram, or open CPG that defines how the primitive systems will be composed. The number of boxes in the composition pattern is the number of primitive systems that will be composed.

3. **Define the primitive systems to compose:**  For an undirected composition pattern the primitive systems are implemented by resource sharers. A resource sharer has four components:
        - ports
        - states
        - a dynamics function $f$
        - a port map $p$
    The dynamics function can be either continuous time $\dot u(t) = f(u(t), p, t)$ or discrete time $u_{n +1} = f(u_n, p, t)$. In both cases $u$ is the state and $p$ is a container for the parameters. Also in both cases, the port map assigns a state to each port. We say the port exposes the state it is assigned. For continuous time use `ContinuousResourceSharer{T}` and for discrete time use `DiscreteResourceSharer{T}`. The type `T` represents the values the states can take on.

    For a directed composition pattern the primitive systems are implemented by machines. A machine has five components:
    - inputs (also called exogenous variables),
    - states
    - outputs
    - a dynamics function $f$
    - a readout function $r$

    The dynamics function can be either continuous time $\dot u(t) = f(u(t), x(t), p, t)$ or discrete time $u_{n+1} = f(u_n, x_n, p, t)$. In both cases $u$ is the state, $x$ contains the exogenous variables, and $p$ contains the parameters. Also in both cases, the readout function $r(u(t))$ is a function of the state. For continuous time use `ContinuousMachine{T}` and for discrete time use `DiscreteMachine{T}`. The type `T` represents the values the inputs, states, and outputs can take on.

4. **Compose:** The `oapply` method takes a composition pattern and  primitive systems. It returns the composite system. Each `oapply` method implements an operad algebras which define to composition of dynamical systems.
    - 

An optional 5th step is to solve and plot the solution to the composite system. 
- For continuous machines or resource sharers, you can construct an `ODEProblem`. Be sure to import [OrdinaryDiffEq](https://diffeq.sciml.ai/stable/tutorials/ode_example/). When the composition pattern is a wiring diagram, we recommend solvers `Tsit5()` with `dtmax` specified and `FRK65(w=0)`.
- For discrete machines or resource sharers, you can construct a `DiscreteDynamicalSystem` or explicitly compute a trajectory using `trajectory`. In either case be sure to import [DynamicalSystems](https://juliadynamics.github.io/DynamicalSystems.jl/latest/).
    

## Continuous Examples: Lotka-Volterra Two Ways

We will give two examples of deriving the Lotka-Volterra equations as a composition of primitive systems. First, we will show it as the composition of resource sharers where the undirected composition pattern is an undirected wiring diagram. Second we will show it as the composition of machines where the directed composition pattern is a wiring diagram.

This example is quite small and you can easiily work it out by hand, but
this approach is based on solid mathematical theory that allows you to automatically
scale to larger and more complex problems. See XXX for examples of more complex systems.

### Lotka-Volterra via undirected composition

A standard Lotka-Volterra predator-prey model is the composition of three resource sharers:

1. a model of rabbit growth --- this resource sharer has dynamics $\dot r(t) = \alpha r(t)$ and one port which exposes the rabbit population.
2. a model of rabbit/fox predation --- this resource sharer has dynamics $$\dot r(t) = -\beta r(t) f(t), \dot f(t) = \gamma r(t)f(t)$$ and two ports which expose the rabbit and fox populations respectively
3. a model of fox population decline --- this resource sharer has dynamics $\dot f(t) = -\delta f(t)$ and one port which exposes the fox population.

However, there are not two independent rabbit populations --- one that grows and one that gets eaten by foxes. Likewise, there are not two independent fox populations --- one that declines and one that feasts on rabbits. To capture these interactions between the trio of resource sharers, we compose them by identifying the exposed rabbit populations and identifying the exposed fox populations. 



```julia
using AlgebraicDynamics
using AlgebraicDynamics.UWDDynam
using Catlab.WiringDiagrams
using OrdinaryDiffEq, Plots

const UWD = UndirectedWiringDiagram

# Define the primitive systems
α, β, γ, δ = 0.3, 0.015, 0.015, 0.7

dotr(u,p,t) = α*u
dotrf(u,p,t) = [-β*u[1]*u[2], γ*u[1]*u[2]]
dotf(u,p,t) = -δ*u

rabbit_growth = ContinuousResourceSharer{Float64}(1, dotr)
rabbitfox_predation = ContinuousResourceSharer{Float64}(2, dotrf)
fox_decline = ContinuousResourceSharer{Float64}(1, dotf)

# Define the composition pattern
rabbitfox_pattern = UWD(2)
add_box!(rabbitfox_pattern, 1); add_box!(rabbitfox_pattern, 2); add_box!(rabbitfox_pattern, 1)
add_junctions!(rabbitfox_pattern, 2)
set_junction!(rabbitfox_pattern, [1,1,2,2]); set_junction!(rabbitfox_pattern, [1,2], outer=true)

# Compose
rabbitfox_system = oapply(rabbitfox_pattern, [rabbit_growth, rabbitfox_predation, fox_decline])

# Solve and plot
u0 = [10.0, 100.0]
tspan = (0.0, 100.0)

prob = ODEProblem(rabbitfox_system, u0, tspan)
sol = solve(prob, Tsit5())

plot(sol, lw=2, title = "Lotka-Volterra Predator-Prey Model", label=["rabbits" "foxes"])
xlabel!("Time")
ylabel!("Population size")
```

![Lotka-Volterra Solutions](/docs/img/lv.svg)

### Lotka-Volterra via directed composition
A standard Lotka-Volterra predator-prey model is the composition of two machines:

1. Evolution of a rabbit population &mdash; this machine has one parameter which represents a population of predators, $h$, that hunt rabbits. This machine has one output which emits the rabbit population $r$. The dynamics of this machine is the parameterized ODE $$\dot r(t) = \alpha r(t) - \beta r(t) h(t).$$ 

2. Evoluation of a fox population &mdash; this machine has one parameter which represents a population of prey, $e$, that are eaten by foxes. This machine has one output which emits the fox population $f$. The dynamics of this machine is the parameterized ODE $$\dot f(t) =\gamma f(t)e(t) - \delta f(t).$$ 

Since foxes hunt rabbit, these machines compose by setting the fox population to be the parameter for rabbit evolution. Likewise, we set the rabbit population to be the parameter for fox evolution. 


```julia
using AlgebraicDynamics
using AlgebraicDynamics.DWDDynam
using Catlab.WiringDiagrams
using OrdinaryDiffEq, Plots

# Define the primitive systems
α, β, γ, δ = 0.3, 0.015, 0.015, 0.7

dotr(x, p, t) = [α*x[1] - β*x[1]*p[1]]
dotf(x, p, t) = [γ*x[1]*p[1] - δ*x[1]]

rabbit = ContinuousMachine{Float64}(1,1,1, dotr, x -> x)
fox    = ContinuousMachine{Float64}(1,1,1, dotf, x -> x)

# Define the composition pattern
rabbitfox_pattern = WiringDiagram([], [])
rabbit_box = add_box!(rabbitfox_pattern, Box(:rabbit, [:pop], [:pop]))
fox_box = add_box!(rabbitfox_pattern, Box(:fox, [:pop], [:pop]))

add_wires!(rabbitfox_pattern, Pair[
    (rabbit_box, 1) => (fox_box, 1),
    (fox_box, 1) => (rabbit_box, 1)
])

# Compose
rabbitfox_system = oapply(rabbitfox_pattern, [rabbit, fox])

# Solve and plot
u0 = [10.0, 100.0]
tspan = (0.0, 100.0)

prob = ODEProblem(rabbitfox_system, u0, tspan)
sol = solve(prob, Tsit5(); dtmax = 0.01)

plot(sol, lw=2, title = "Lotka-Volterra Predator-Prey Model", label=["rabbits" "foxes"])
xlabel!("Time")
ylabel!("Population size")
```
![Lotka-Volterra Solutions](/docs/img/lv2.svg)


## Discrete Example: Cellular automata

We will derive a cellular automaton as a composition of primitive machines using the Open CPGs as our composition syntax. The composition pattern will be a row of $n$ cells each of which is connected to its two neighbors. A cell both sends its state to its neighbors and receiving its neighbors' states. The primitive systems are identical machines whose dicrete dynamics implement a specified rule.

```julia
using AlgebraicDynamics.DWDDynam
using AlgebraicDynamics.CPortGraphDynam
using AlgebraicDynamics.CPortGraphDynam: gridpath

using Catlab.CategoricalAlgebra
using Catlab.WiringDiagrams
using Catlab.WiringDiagrams.CPortGraphs

using DynamicalSystems, Plots

function Rule(k::Int)
    (left_neighbor, x, right_neighbor) -> 
    Bool(digits(k, base=2, pad=8)[1 + right_neighbor + 2*x + 4*left_neighbor])
end

# Define the composition pattern
n = 100
row = apex(gridpath(n, 1))

# Define the primitive system
rule = DiscreteMachine{Bool}(2, 1, 2, (u, x, p, t)->Rule(p)(x[2], u[1], x[1]), 
            u->[u[1], u[1]])

# Compose
automaton = oapply(row, rule)

## Solve and plot
u0 = zeros(Int, n); u0[Int(n/2)] = 1

rule_number = 126
traj = trajectory(automaton, u0, [0,0], rule_number, 100)
spy(Matrix(traj))
```
![Cellular Automaton Solutions](/docs/img/rule126.svg)

## Future Work
- Add more integrators beyond the simple Euler's method.
- Include higher order polynomial methods and simplectic and implicit methods for physical problems. 
- Integrate with [AlgebraicPetri.jl](https://algebraicjulia.github.io/AlgebraicPetri.jl/dev/)
