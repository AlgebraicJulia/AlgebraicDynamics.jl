using Catlab.Graphics
using Catlab.CategoricalAlgebra
using Catlab.Programs.RelationalPrograms

using Distributions
using Plots

using AlgebraicDynamics.DiscDynam

function get_coords(nesting::Int)
  state = zeros(Int, 2^nesting, 2^nesting)
  _get_coord_map!(state, nesting-1, [0,0], 0)
  return state
end
function _get_coord_map!(state::Array{Int, 2}, nesting::Int, offset::Array{Int, 1}, cur_val::Int)
  if nesting == 0
    state[(offset + [1,1])...] = cur_val + 1
    state[(offset + [2,1])...] = cur_val + 2
    state[(offset + [1,2])...] = cur_val + 3
    state[(offset + [2,2])...] = cur_val + 4
    return true
  end
  scale = 2^nesting

  nw = _get_coord_map!(state, nesting - 1, offset+[0,0]*scale,cur_val)
  ne = _get_coord_map!(state, nesting - 1, offset+[1,0]*scale,cur_val + scale^2)
  sw = _get_coord_map!(state, nesting - 1, offset+[0,1]*scale,cur_val + 2*scale^2)
  se = _get_coord_map!(state, nesting - 1, offset+[1,1]*scale,cur_val + 3*scale^2)
  return true
end
function draw_dynam(dynam; kw...)
  uwd = UntypedRelationDiagram{Symbol, Symbol}()
  copy_parts!(uwd, dynam)
  to_graphviz(uwd; kw...)
end

function initialize_dot(center::Array{<:Real, 1}, std::Array{<:Real, 1}, nesting::Int)
  width = 2^nesting
  # Initialize center values in stencils
  state = zeros(width, width)
  dist = Distributions.MvNormal(center, std)
  for x in 1:width
    for y in 1:width
      state[x,y] = pdf(dist, [(x-1)/(width-1), (y-1)/(width-1)])
    end
  end
  # Interpolate port values
  val_to_state = get_coords(nesting)*5
  values = interp_vals(state, val_to_state)
  return values, val_to_state
end

function interp_vals(values::Array{<:Real, 2}, val_to_state::Array{Int,2})
  width = size(values)[1]
  res = zeros(width*width*5)
  res[val_to_state] = values
  for x in 1:width
    for y in 1:width
      cur_ind = val_to_state[x,y] - 4
      res[cur_ind:(cur_ind+3)] .= res[cur_ind+4]
      #res[cur_ind:(cur_ind+3)] .= 0
      if x + 1 <= width
        e_ind = val_to_state[x+1,y]
        res[cur_ind] = (res[cur_ind] + res[e_ind])/2
      end
      if y - 1 > 0
        s_ind = val_to_state[x,y-1]
        res[cur_ind+1] = (res[cur_ind+1] + res[s_ind])/2
      end
      if x - 1 > 0
        w_ind = val_to_state[x-1,y]
        res[cur_ind+2] = (res[cur_ind+2] + res[w_ind])/2
      end
      if y + 1 <= width
        n_ind = val_to_state[x,y+1]
        res[cur_ind+3] = (res[cur_ind+3] + res[n_ind])/2
      end
    end
  end
  return res
end

function initialize_dot(center::Array{<:Real, 1}, std::Array{<:Real, 1}, nesting::Int, Δx::Real)
  width = 2^nesting
  # Initialize center values in stencils
  state = zeros(width, width)
  dist = Distributions.MvNormal(center, std)
  for x in 1:width
    for y in 1:width
      state[x,y] = pdf(dist, [(x-1)/(width-1), (y-1)/(width-1)])
    end
  end
  # Interpolate port values
  val_to_state = get_coords(nesting)*6 .- 1
  values = interp_vals(state, val_to_state, Δx)
  return values, val_to_state
end

function interp_vals(values::Array{<:Real, 2}, val_to_state::Array{Int,2}, Δx::Real)
  width = size(values)[1]
  res = zeros(width*width*6)
  res[val_to_state] = values
  for x in 1:width
    for y in 1:width
      cur_ind = val_to_state[x,y] - 4
      res[cur_ind+5] = (y-1)*Δx
      res[cur_ind:(cur_ind+3)] .= res[cur_ind+4]
      #res[cur_ind:(cur_ind+3)] .= 0
      if x + 1 <= width
        e_ind = val_to_state[x+1,y]
        res[cur_ind] = (res[cur_ind] + res[e_ind])/2
      end
      if y - 1 > 0
        s_ind = val_to_state[x,y-1]
        res[cur_ind+1] = (res[cur_ind+1] + res[s_ind])/2
      end
      if x - 1 > 0
        w_ind = val_to_state[x-1,y]
        res[cur_ind+2] = (res[cur_ind+2] + res[w_ind])/2
      end
      if y + 1 <= width
        n_ind = val_to_state[x,y+1]
        res[cur_ind+3] = (res[cur_ind+3] + res[n_ind])/2
      end
    end
  end
  return res
end
