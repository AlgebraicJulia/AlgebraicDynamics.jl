module Steady

export Machine, ss

"""
    Machine

A discrete dynamical system is a quintuple ``(I, O, S, u, r)``, where

  - ``I`` is a set of inputs
  - ``O`` is a set of outputs
  - ``S`` is a set of states
  - ``u: I \\times S \\to S`` is an update function
  - ``r: S \\to O`` is a readout function

"""
struct Machine
    """
        Uᵢⱼ = u(j, i)
    """
    U::Matrix{Int}
    """
        Rⱼ = r(j)
    """
    R::Vector{Int}
end

"""
    ss(machine::Machine)

Construct a matrix
```math
    S: O \\times I \\to \\mathbb{N}
```
whose entry ``S_{ij}`` contains the number of states ``s \\in S`` 
such that

  - ``u(j, s) = s`` and
  - ``r(s) = i``.

"""
function ss(machine::Machine)
    # update matrix
    U = machine.U

    # readout matrix
    R = machine.R

    # steady-state matrix
    S = zeros(size(U))

    for j in axes(U, 2)
        for i in axes(U, 1)
            k = R[i]
            
            if U[i, j] == i
                # i is a (j, k) steady-state
                S[k, j] += 1
            end
        end
    end

    return S
end

end
