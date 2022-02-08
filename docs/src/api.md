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

## Integration with [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/#Problem-Types)
```@docs
AlgebraicDynamics.UWDDynam.ODEProblem
AlgebraicDynamics.UWDDynam.DDEProblem
AlgebraicDynamics.UWDDynam.DiscreteProblem
```
