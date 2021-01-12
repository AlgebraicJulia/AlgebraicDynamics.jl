# # [Ecosystem Models](@id ecosystem_example)
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/examples/ecosystem.ipynb)

using AlgebraicDynamics
using AlgebraicDynamics.DWDDynam
using AlgebraicDynamics.UWDDynam

using Catlab.CategoricalAlgebra
using Catlab.WiringDiagrams
const UWD = UndirectedWiringDiagram

using OrdinaryDiffEq
using Plots, Plots.PlotMeasures

# ## Land Ecosystem

# ### Rabbits and foxes

# A standard Lotka Volterra predator-prey model is the composition of three primitive resource sharers:

# 1. a model of rabbit growth: this resource sharer has dynamics $\dot r(t) = \alpha r(t)$ and one port which exposes the rabbit population.
# 2. a model of rabbit/fox predation: this resource sharer has dynamics $$\dot r(t) = -\beta r(t) f(t), \dot f(t) = \gamma r(t)f(t)$$ and two ports which expose the rabbit and fox populations respectively.
# 3. a model of fox population decline: this resource sharer has dynamics $\dot f(t) = -\delta f(t)$ and one port which exposes the fox population.

# However, there are not two independent rabbit populations -- one that grows and one that gets eaten by foxes. Likewise, there are not two independent fox populations -- one that declines and one that feasts on rabbits. To capture these interactions between the trio of resource sharers, we compose them by identifying the exposed rabbit populations and identifying the exposed fox populations. 
# The syntax for this undirected composition is defined by an undirected wiring diagram.


## Define the primitive systems
dotr(u,p,t) = p[1]*u
dotrf(u,p,t) = [-p[2]*u[1]*u[2], p[3]*u[1]*u[2]]
dotf(u,p,t) = -p[4]*u

rabbit_growth = ContinuousResourceSharer{Float64}(1, dotr)
rabbitfox_predation = ContinuousResourceSharer{Float64}(2, dotrf)
fox_decline = ContinuousResourceSharer{Float64}(1, dotf)

## Define the composition pattern
rabbitfox_pattern = UWD(2)
add_box!(rabbitfox_pattern, 1); add_box!(rabbitfox_pattern, 2); add_box!(rabbitfox_pattern, 1)
add_junctions!(rabbitfox_pattern, 2)
set_junction!(rabbitfox_pattern, [1,1,2,2]); set_junction!(rabbitfox_pattern, [1,2], outer=true)

## Compose
rabbitfox_system = oapply(rabbitfox_pattern, [rabbit_growth, rabbitfox_predation, fox_decline])

# We can now construct an `ODEProblem` from the resource sharer `rabbitfox_system` and plot the solution.

α, β, γ, δ = 0.3, 0.015, 0.015, 0.7
params = [α, β, γ, δ]

u0 = [10.0, 100.0]
tspan = (0.0, 100.0)

prob = ODEProblem(rabbitfox_system, u0, tspan, params)
sol = solve(prob, Tsit5())

plot(sol, lw=2, title = "Lotka-Volterra Predator-Prey Model", bottom_margin=10mm, left_margin=10mm, label=["rabbits" "foxes"])
xlabel!("Time")
ylabel!("Population size")

# ### Rabbits, foxes, and hawks
# Suppose we now have a three species ecosystem containing rabbits, foxes, and hawks. Foxes and hawks both prey upon rabbits but do not interact with each other. This ecosystem consists of five primitive systems which share variables.
# 1. rabbit growth:  $\dot r(t) = \alpha r(t)$
# 2. rabbit/fox predation:  $\dot r(t) = -\beta r(t) f(t), \dot f(t) = \delta r(t)f(t)$
# 3. fox decline:  $\dot f(t) = -\gamma f(t)$
# 4. rabbit/hawk predation: $\dot r(t) = -\beta' r(t)h(t), \dot h(t) = \delta' r(t)h(t)$
# 5. hawk decline:  $\dot h(t) = -\gamma' h(t)$

# This means the desired composition pattern has five boxes and many ports and wires to keep track of. Instead of implementing this composition pattern by hand, we construct it as a pushout.


## Define the composition pattern for rabbit growth
rabbit_pattern = UWD(1)
add_box!(rabbit_pattern, 1); add_junctions!(rabbit_pattern, 1)
set_junction!(rabbit_pattern, [1]); set_junction!(rabbit_pattern, [1], outer=true)

## Define the composition pattern for the rabbit/hawk Lotka-Volterra model
rabbithawk_pattern = rabbitfox_pattern

## Define transformations between the composition patterns
rabbitfox_transform  = ACSetTransformation((Box=[1], Junction=[1], Port=[1], OuterPort=[1]), rabbit_pattern, rabbitfox_pattern)
rabbithawk_transform = ACSetTransformation((Box=[1], Junction=[1], Port=[1], OuterPort=[1]), rabbit_pattern, rabbithawk_pattern)

## Take the pushout to define the composition pattern for the rabbit, fox, hawk system
rabbitfoxhawk_pattern = ob(pushout(rabbitfox_transform, rabbithawk_transform))

#-
## Define the additional primitive systems
dotrh(x, p, t) = [-p[5]*x[1]*x[2], p[6]*x[1]*x[2]]
doth(x, p, t)  = -p[7]*x

rabbithawk_predation = ContinuousResourceSharer{Float64}(2, dotrh)
hawk_decline         = ContinuousResourceSharer{Float64}(1, doth)

## Compose
land_system = oapply(rabbitfoxhawk_pattern, 
                        [rabbit_growth, rabbitfox_predation, fox_decline, 
                         rabbithawk_predation, hawk_decline])

## Solve and plot
β′, γ′, δ′ = .01, .01, .5
params = vcat(params, [β′, γ′, δ′])

u0 = [10.0, 100.0, 50.0]
tspan = (0.0, 100.0)

prob = ODEProblem(land_system, u0, tspan, params)
sol = solve(prob, Tsit5())

plot(sol, lw=2, title = "Land Ecosystem", bottom_margin=10mm, left_margin=10mm, label=["rabbits" "foxes" "hawks"])
xlabel!("Time")
ylabel!("Population size")

# Unfortunately, the hawks are going extinct in this model. We'll have to give hawks something else to eat!
#-
# ## Ocean Ecosystem

# Consider a ocean ecosystem containing three species —- little fish, big fish, and sharks -— with two predation interactions —- sharks eat big fish and big fish eat little fish.

# This ecosystem can be modeled as the composition of 3 machines:
# 1. Evolution of the little fish population:  this machine has one exogenous variable which represents a population of predators $h(t)$ that hunt little fish. This machine has one output which emits the little fish population. The dynamics of this machine is the driven ODE $$\dot f(t) = \alpha f(t) - \beta f(t)h(t)$$
# 2. Evolution of the big fish population:  this machine has two exogenous variables which represent a population of prey $e(t)$ that are eaten by big fish and a population of predators $h(t)$ which hunt big fish. This machine has one output which emits the big fish population. The dynamics of this machine is the drive ODE $$\dot F(t) = \gamma F(t)e(t) - \delta F(t) - \beta'F(t)h(t)$$
# 3. Evolution of the shark population:  this machine has one exogenous variable which represents a population of prey $e(t)$ that are eaten by sharks. This machine has one output which emits the shark population. The dynamics of this machine is the driven ODE $$\dot s(t) = \gamma's(t)e(t) - \delta's(t)$$


## Define the primitive systems
dotfish(f, x, p, t) = [p[1]*f[1] - p[2]*x[1]*f[1]]
dotFISH(F, x, p, t) = [p[3]*x[1]*F[1] - p[4]*F[1] - p[5]*x[2]*F[1]]
dotsharks(s, x, p, t) = [p[6]*s[1]*x[1]-p[7]*s[1]]

fish   = ContinuousMachine{Float64}(1,1,1, dotfish,   f->f)
FISH   = ContinuousMachine{Float64}(2,1,1, dotFISH,   F->F)
sharks = ContinuousMachine{Float64}(1,1,1, dotsharks, s->s)

# We compose these machines by (1) sending the output of the big fish machine as the input to both the little fish and shark machines and (2) sending the output of the little fish and shark machines as the inputs to the big fish machine.
# The syntax for this directed composition is given by a directed wiring diagram.

## Define the composition pattern
ocean_pattern = WiringDiagram([], [])
fish_box = add_box!(ocean_pattern, Box(:fish, [:pop], [:pop]))
Fish_box = add_box!(ocean_pattern, Box(:Fish, [:pop, :pop], [:pop]))
shark_box = add_box!(ocean_pattern, Box(:shark, [:pop], [:pop]))

add_wires!(ocean_pattern, Pair[
    (fish_box, 1)  => (Fish_box, 1),
    (shark_box, 1) => (Fish_box, 2),
    (Fish_box, 1)  => (fish_box, 1),
    (Fish_box, 1)  => (shark_box, 1)
])

## Compose
ocean_system = oapply(ocean_pattern, [fish, FISH, sharks])

## Solve and plot
α, β, γ, δ, β′, γ′, δ′ = 0.35, 0.015, 0.015, 0.7, 0.017, 0.017, 0.35
params = [α, β, γ, δ, β′, γ′, δ′]

u0 = [100.0, 10, 2.0]
tspan = (0.0, 100.0)

prob = ODEProblem(ocean_system, u0, tspan, params)
sol = solve(prob, FRK65(0))

plot(sol, lw=2, title = "Ocean Ecosystem", bottom_margin=10mm, left_margin=10mm, label=["little fish" "big fish" "sharks"])
xlabel!("Time")
ylabel!("Population size")

# ## Total ecosystem
# ### Another layer of composition

# We will introduce a final predation interaction -- hawks eat little fish --  which will combine the land and ocean ecosystems.

# There will be 16 parameters in to the total ecosystem.
# - parameters 1-7 will determine the land ecosystem
# - parameters 8 and 9 will determine the hawk/little fish predation. Parameter 8 gives the rate of hawk growth and parameter 8 gives the rate of little fish decline.
# - parameter 10-16 will determine the ocean ecosystem.

# The composition will be as resource shareres so the first thing we will do is use the dynamics of the machine `ocean_system` to define the dynamics of a resource sharer. We will also define a resource sharer that models hawk/little fish predation.


## Define the additional primitive systems
ocean_system_rs = ContinuousResourceSharer{Float64}(3, (u,p,t)->eval_dynamics(ocean_system, u, [], p[10:16]))

dothf(u,p,t) = [p[8]*u[1]*u[2], -p[9]*u[1]*u[2]]
fishhawk_predation = ContinuousResourceSharer{Float64}(2, dothf)

## Define the composition pattern
eco_pattern = UWD(0)
add_box!(eco_pattern, 3); add_box!(eco_pattern, 2); add_box!(eco_pattern, 3)
add_junctions!(eco_pattern, 6)
set_junction!(eco_pattern, [1,2,3,3,4,4,5,6])

## Compose
eco_system=oapply(eco_pattern, [land_system, fishhawk_predation, ocean_system_rs])

# We can now plot the evolution of the total ecosystem.

## Solve and plot
u0 = [100.0, 50.0, 20.0, 100, 10, 2.0]
tspan = (0.0, 100.0)

params = [0.3, 0.015, 0.015, 0.7, .01, .01, .5, 
          0.001, 0.003, 
          0.35, 0.015, 0.015, 0.7, 0.017, 0.017, 0.35]

prob = ODEProblem(eco_system, u0, tspan, params)
sol = solve(prob, Tsit5())
plot(sol, lw=2, label = ["rabbits" "foxes" "hawks" "little fish" "big fish" "sharks"])

# Let's zoom in on a narrower time-window.
tspan = (0.0, 30.0)

prob = ODEProblem(eco_system, u0, tspan, params)
sol = solve(prob, Tsit5())
plot(sol, lw=2, label = ["rabbits" "foxes" "hawks" "little fish" "big fish" "sharks"])

# As a sanity check we can define the rates for the hawk/little fish predation to be 0. This decouples the land and ocean ecosystems. As expected, the plot shows the original evolution of the land ecosystem overlayed with the original evolution of the ocean ecosystem. This shows that they two ecosystems now evolve independently.

tspan = (0.0, 100.0)
params = [0.3, 0.015, 0.015, 0.7, .01, .01, .5, 
          0, 0, 
          0.35, 0.015, 0.015, 0.7, 0.017, 0.017, 0.35]
prob = ODEProblem(eco_system, u0, tspan, params)
sol = solve(prob, Tsit5())
plot(sol, lw=2, label = ["rabbits" "foxes" "hawks" "little fish" "big fish" "sharks"])




