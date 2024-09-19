module AlgebraicDynamicsPlotsExt

using AlgebraicDynamics

using Plots

### Plotting backend
@recipe function f(sol, m::AbstractMachine, p=nothing)
    labels = (String ∘ Symbol).(output_ports(m))
    label --> reshape(labels, 1, length(labels))
    vars --> map(1:noutputs(m)) do i
        ((t, args...) -> (t, readout(m)(collect(args), p, t)[i]), 0:nstates(m)...)
    end
    sol
end

### Plotting backend
@recipe function f(sol, r::AbstractResourceSharer)
    labels = (String ∘ Symbol).(collect(view(ports(r), portmap(r))))
    label --> reshape(labels, 1, length(labels))
    vars --> portmap(r)
    sol
end

end