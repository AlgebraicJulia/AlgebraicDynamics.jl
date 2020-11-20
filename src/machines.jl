module Machines

using Catlab.WiringDiagrams

export RealMachine, oapply

struct RealMachine{T,U}
  parameters::Int
  states::Int
  outputs::Int
  update::T
  readout::U
end


function oapply(composite::WiringDiagram, dynamics::Vector{RealMachine{Function, Function}})
    nboxes(composite) == length(dynamics)  || error("there are $nboxes(composite) boxes but $length(dynamics) machines")

    for b in 1:nboxes(composite)
        params = dynamics[b].parameters 
        in_ports = length(input_ports(composite, box_ids(composite)[b])) 
        params == in_ports || error("there are $in_ports input ports for box $b but $params parameters for machine $b")

        outputs = dynamics[b].outputs 
        out_ports = length(output_ports(composite, box_ids(composite)[b])) 
        outputs == out_ports || error("there are $out_ports output ports for box $b but $outputs outputs for machine $b")
    end

    parameters = length(output_ports(composite, input_id(composite)))
    outputs = length(input_ports(composite, output_id(composite)))
    states = sum(d -> d.states, dynamics)
    
    state_ids = zeros(Int64, length(dynamics) + 1)
    state_ids[1] = 1
    for i in 1:length(dynamics)
        state_ids[i+1] = state_ids[i] + dynamics[i].states
    end

    function internal_readout(p,x)
        r = Dict(input_id(composite) => p) # dictionary indexed by box id and values - the state one the output ports of the box
        for b in 1:nboxes(composite)

            r[box_ids(composite)[b]] = dynamics[b].readout(view(x, state_ids[b]:(state_ids[b + 1] - 1)))
        end
        return r
    end

    function update(p, x, args...)
        r = internal_readout(p,x)

        dx = zero(x)
        for b in 1:nboxes(composite)

            params = zeros(dynamics[b].parameters)
            for w in in_wires(composite, box_ids(composite)[b]) # get incoming wires
                params[w.target.port] = r[w.source.box][w.source.port]
            end

            state_idx = state_ids[b]:(state_ids[b + 1] - 1)
            dx[state_idx] += dynamics[b].update(params,  view(x, state_idx), args...)
        end
        return dx
    end

    function readout(x)
        out = zeros(outputs)
        p = zeros(parameters)
        r = internal_readout(p, x)
        for w in in_wires(composite, output_id(composite))
            out[w.target.port] += r[w.source.box][w.source.port]
        end
        return out
    end
    
    return RealMachine{Function, Function}(parameters, states, outputs, update, readout)
end

end #module