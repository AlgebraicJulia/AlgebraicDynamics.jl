# AlgebraicDynamics.jl

AlgebraicDynamics.jl is a library for compositional dynamical systems. We build on [Catlab.jl](https://algebraicjulia.github.io/Catlab.jl/dev/) to provide a software interface for specifying and solving dynamical systems with compositional and hierarchical structure. The implementation of the composition of dynamical systems follows the mathematics of operads and operad algebras.


## Composing Dynamical Systems

The general process for composing  dynamical systems is as follows:

1. __Pick a composition syntax.__ A composition syntax may be either undirected or directed. In this library, composition is implemented for the following syntaxes:
    - undirected wiring diagrams (an undirected composition syntax)
    - wiring diagrams (a directed composition syntax)
    - open circular port graphs, also called open CPGs (a directed composition syntax)

2. __Define a composition pattern.__ Implement a specific undirected wiring diagram, wiring diagram, or open CPG that defines how the primitive systems will be composed. The number of boxes in the composition pattern is the number of primitive systems that will be composed.

3. __Define the primitive systems to compose.__
   - For an undirected composition pattern, the primitive systems are implemented by resource sharers. A resource sharer has four components:
     - ports
     - states
     - a dynamics function, ``f``
     - a port map,  ``m``
     The dynamics function can be either continuous time ``\dot u(t) = f(u(t), p, t)`` or discrete time ``u_{n +1} = f(u_n, p, t)``. In both cases, ``u`` contains the state and ``p`` contains the parameters. Also in both cases, the port map assigns a state to each port. We say the port exposes the state it is assigned. For continuous time use `ContinuousResourceSharer{T}` and for discrete time use `DiscreteResourceSharer{T}`. The type `T` represents the values that the states can take on.

   - For a directed composition pattern, the primitive systems are implemented by machines. A machine has five components:
     - inputs (also called exogenous variables)
     - states
     - outputs
     - a dynamics function,  ``f``
     - a readout function,  ``r``
     The dynamics function can be either continuous time ``\dot u(t) = f(u(t), x(t), p, t)`` or discrete time ``u_{n+1} = f(u_n, x_n, p, t)``. In both cases, ``u`` contains the state, ``x`` contains the exogenous variables, and ``p`` contains the parameters. Also in both cases, the readout function ``r(u(t))`` is a function of the state. For continuous time use `ContinuousMachine{T}` and for discrete time use `DiscreteMachine{T}`. The type `T` represents the values that the inputs, states, and outputs can take on.

4. __Compose.__ The `oapply` method takes a composition pattern and  primitive systems, and it returns the composite system. Each `oapply` method implements an operad algebra which specifies a regime for composing dynamical systems. See [[Schultz et al. 2019](https://arxiv.org/abs/1609.08086)] and [[Vagner et al. 2015](https://arxiv.org/abs/1408.1598)] for definitions of the operad algebras ``\mathsf{CDS}`` and ``\mathsf{DDS}`` for directed composition. See [[Baez and Pollard 2017](https://arxiv.org/abs/1704.02051)] for definitions of the operad algebra ``\mathsf{Dynam}`` for undirected composition. See [[Libkind 2020](https://arxiv.org/abs/2007.14442)] for a general overview of these operad algebras.

## Algebraic Approaches to Defining Primitives

While the operadic approach to composing hierarchical dynamical systems is the primary feature set of this package, many families of dynamical systems have algebraic or combinatorial specifications.

Some examples include:

1. The mass action kinetics of a Petri Net defined in [AlgebraicPetri.jl](https://github.com/AlgebraicJulia/AlgebraicPetri.jl) [#126](https://github.com/AlgebraicJulia/AlgebraicDynamics.jl/pull/126)
2. The Lotka-Volterra dynamics of a [Regulatory Network](https://github.com/AlgebraicJulia/RegNets.jl) [#103](@ref)
3. [Threshold Linear Networks](@ref), which are a model of biological neurons in the brain [#136](https://github.com/AlgebraicJulia/AlgebraicDynamics.jl/pull/103)

## Solving Dynamical Systems

Once you have built the composite system, you can solve it and plot the solution.
- For continuous machines and resource sharers, you can construct an
  [`ODEProblem`](https://diffeq.sciml.ai/stable/tutorials/ode_example/) or a
  [`DDEProblem`](https://diffeq.sciml.ai/stable/tutorials/dde_example/).
  When the composition pattern is a wiring diagram, we recommend solvers
  `Tsit5()` with `dtmax` specified and `FRK65(w=0)`.
- For discrete machines and resource sharers, you can construct a
  [`DiscreteProblem`](https://diffeq.sciml.ai/stable/types/discrete_types/)
  or explicitly compute a `trajectory`.
In either case, be sure to import [OrdinaryDiffEq.jl](https://diffeq.sciml.ai/latest/).


## Future Work
- Discover more functorial integrators beyond the simple Euler's method
- Include higher order polynomial methods and symplectic and implicit methods for physical problems
- Integrate with [AlgebraicPetri.jl](https://algebraicjulia.github.io/AlgebraicPetri.jl/dev/)
