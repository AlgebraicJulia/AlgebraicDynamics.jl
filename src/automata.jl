using Catlab
using Catlab.Doctrines
using RecursiveArrayTools

export observer, observel, initial, next, nsteps

const FreeSMC = FreeSymmetricMonoidalCategory
# store states as lists of nested pairs
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
    g = composite.args[2]
    return observer(g, curr[2])
end

function observer(product::FreeSMC.Hom{:otimes}, curr)
    f,g = product.args[1], product.args[2]
    return(observer(f, curr[1]), observer(g, curr[2]))
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
    f,g = product.args[1], product.args[2]
    return(observel(f, curr[1]), observel(g, curr[2]))
end

"""    initial(f)

list out the initial state of the system
"""
function initial(f::FreeSMC.Hom{:generator})
    return f.args[1].states
end

function initial(composite::FreeSMC.Hom{:compose})
    f,g = composite.args[1], composite.args[2]
    fstates = initial(f)
    gstates = initial(g)

    states = [(fs, gs) for fs=fstates, gs=gstates if observer(f, fs)==observel(g, gs)]
    return states
end

function initial(product::FreeSMC.Hom{:otimes})
    f,g = product.args[1], product.args[2]
    fstates = initial(f)
    gstates = initial(g)

    states = [(fs, gs) for fs=fstates, gs=gstates if true] #the "if true" flattens the array
    return states
end

"""    next(f, curr)

compute the set of possible next states of the system. This is the core of
nondeterministic evolution of the automata.
"""
function next(sys::FreeSMC.Hom{:generator}, curr)
    f = sys.args[1]
    nexts = []
    for i=curr
        A = f.f[:, i] #grab the ith column
        append!(nexts, findall(x->x==1, A)) # append indices of rows showing a 1
    end
    return unique(nexts)
end

function next(sys::FreeSMC.Hom{:id}, curr)
    return curr
end

function next(composite::FreeSMC.Hom{:compose}, curr)
    nexts = []
    f, g = composite.args[1], composite.args[2]

    for (curr1, curr2)=curr

        fstates = next(f,curr1)
        gstates = next(g,curr2)

        append!(nexts,[(fs, gs) for fs=fstates, gs=gstates if observer(f, fs)==observel(g, gs)])
        end
    return nexts #don't unique because that would be very expensive
end



function next(product::FreeSMC.Hom{:otimes}, curr)
    nexts = []
    f, g = product.args[1], product.args[2]

    for (curr1, curr2)=curr

        fstates = next(f,curr1)
        gstates = next(g,curr2)

        append!(nexts,[(fs, gs) for fs=fstates, gs=gstates if true])
        end
    return nexts #don't unique because that would be very expensive
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
