using Catlab
using Catlab.Doctrines

export observer, observel, initial, next, nsteps

const FreeSMC = FreeSymmetricMonoidalCategory


"""    observer(f, curr)

maps a set of states to a set of possible observations
on the right side of the automata.
"""
function observer(f::FreeSMC.Hom{:generator}, curr)
    return f.args[1].outputs[curr]
end

function observer(f::FreeSMC.Hom{:id}, curr)
    return curr
end

function observer(composite::FreeSMC.Hom{:compose}, curr)
    g = last(composite.args)
    return observer(g, last(curr))
end

function observer(product::FreeSMC.Hom{:otimes}, curr)
    return map(observer, product.args, curr)
end


"""    observel(f, curr)

maps a set of states to a set of possible observations
on the left side of the automata.
"""
function observel(f::FreeSMC.Hom{:generator}, curr)
    return f.args[1].inputs[curr]
end

function observel(f::FreeSMC.Hom{:id}, curr)
    return curr
end

function observel(composite::FreeSMC.Hom{:compose}, curr)
    f = composite.args[1]
    return observel(f, curr[1])
end

function observel(product::FreeSMC.Hom{:otimes}, curr)
    return map(observel, product.args, curr)
end

""" processStates(states, fs, align)

maps a set of states [s1,..., sn] where each si is a vector of states
corresponding to the ith morphism of fs, to a set of states [t1,...,tk]
where ti represent a state of the total system. In paricular each ti has n components.

When align is set to true states of the total system must match on observation.
"""
function processStates(states, fs, align)
    n = length(states)
    if n==1
        return broadcast(x->Any[x], states[1])
    end

    states1 = processStates(states[1:n-1], fs[1:n-1],  align)
    states2 = states[n]
    totalStates = []
    for s1=states1
        for s2=states2
            if !align || observer(fs[n-1], s1[n-1]) == observel(fs[n], s2)
                push!(totalStates, push!(copy(s1), s2))
            end
        end
    end
    return totalStates
end

"""    initial(f)

list out the initial state of the system
"""
function initial(f::FreeSMC.Hom{:generator})
    return f.args[1].states
end

function initial(composite::FreeSMC.Hom{:compose})
    states = broadcast(initial, composite.args)
    return processStates(states, composite.args, true)
end


function initial(product::FreeSMC.Hom{:otimes})
    states = broadcast(initial, product.args)
    return processStates(states, product.args, false)
end

"""    next(f, currs)

compute the set of possible next states of the system given a vector of
current states.
This is the core of nondeterministic evolution of the automata.
"""
function next(sys::FreeSMC.Hom{:generator}, currs)
    nexts = []
    for curr=currs
        append!(nexts, next_single_state(sys, curr))
    end
    return nexts
end


function next(composite::FreeSMC.Hom{:compose}, currs)
    nexts = []
    for curr=currs
        append!(nexts, next_single_state(composite, curr))
    end
    return nexts
end

function next(product::FreeSMC.Hom{:otimes}, currs)
    nexts = []
    for curr=currs
        append!(nexts, next_single_state(product, curr))
    end
    return nexts
end


"""    next_single_state(f, curr)

compute the set of possible next states of the system given a states curr.
"""

function next_single_state(sys::FreeSMC.Hom{:generator}, curr)
    f = sys.args[1]
    A = f.f[:, curr] #grab the ith column
    return findall(x->x==1, A) # append indices of rows showing a 1
end

function next_single_state(composite::FreeSMC.Hom{:compose}, curr)
    states = broadcast(next_single_state, composite.args, curr)
    return processStates(states, composite.args, true)
end

function next_single_state(product::FreeSMC.Hom{:otimes}, curr)
    states = broadcast(next_single_state, product.args, curr)
    return processStates(states, product.args, false)
end



"""    nsteps(f, n::Int)

evolve the automata system by n time steps, this method returns a Vector of
possible states for the system.
"""
function nsteps(f, n::Int)
    x = initial(f)
    for i in 2:n
        x = next(f, x)
    end
    return x
end

"""    nsteps(f, n::UnitRange)

compute the next steps for each element of the range, this method returns a
Vector of Vectors of possible states where each element is a time step.
"""
function nsteps(f, n::UnitRange)
    x = [initial(f)]
    for i in n[2:end]
        push!(x, next(f, x[end]))
    end
    return x
end
