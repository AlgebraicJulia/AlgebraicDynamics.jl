# Library Reference

## Types of Dynamical Systems

### Machines

```@docs
AlgebraicDynamics.DWDDynam.AbstractMachine
AlgebraicDynamics.DWDDynam.ContinuousMachine
AlgebraicDynamics.DWDDynam.DelayMachine
AlgebraicDynamics.DWDDynam.DiscreteMachine
```

### Resource Sharers

```@docs
AlgebraicDynamics.UWDDynam.AbstractResourceSharer
AlgebraicDynamics.UWDDynam.ContinuousResourceSharer
AlgebraicDynamics.UWDDynam.DelayResourceSharer
AlgebraicDynamics.UWDDynam.DiscreteResourceSharer
```

## Composition of Dynamical Systems

### Operad Algebras

```@docs
AlgebraicDynamics.UWDDynam.oapply
```

### Checks

```@docs
AlgebraicDynamics.UWDDynam.fills
AlgebraicDynamics.DWDDynam.fills
```

## Time Evolution

### Instantaneous Dynamics

```@docs
AlgebraicDynamics.UWDDynam.eval_dynamics
```

### Time Discretization

```@docs
AlgebraicDynamics.UWDDynam.euler_approx
```

### Discrete Trajectory

```@docs
AlgebraicDynamics.UWDDynam.trajectory
```

## Package Extensions

### [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)

```@docs
OrdinaryDiffEq.ODEProblem
OrdinaryDiffEq.DiscreteProblem
```

### [DelayDiffEq.jl](https://github.com/SciML/DelayDiffEq.jl)

```@docs
DelayDiffEq.DDEProblem
```
