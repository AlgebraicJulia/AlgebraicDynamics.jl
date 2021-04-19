""" Tonti Diagrams
This file includes a framework for constructing and evaluating Tonti diagrams.
Tonti diagrams are stored as ACSets, and have an imperative interface for
describing physical variables and the relationships between them. This tooling
also lets a Tonti diagram be converted to a vectorfield, allowing for
simulation of physical systems through any traditional vectorfield solver.
"""

module TontiDiagrams
  using Catlab.Present
  using Catlab.Theories
  using Catlab.CategoricalAlgebra
  using Catlab.CategoricalAlgebra.FinSets
  using Catlab.Graphs
  import Catlab.Graphs: Graph
  using Catlab.Programs
  using Catlab.WiringDiagrams
  using CombinatorialSpaces
  using CombinatorialSpaces: ⋆

  export TontiDiagram, Space, add_variables!, add_derivatives!, add_time_dep!,
          add_laplacian!, add_transition!, add_derivative!, add_variable!, add_bc!,
          vectorfield

  # Rendered as: https://q.uiver.app/?q=WzAsMTAsWzEsMiwiTyJdLFsyLDIsIkkiXSxbMSwxLCJWIl0sWzEsMywiVCJdLFswLDIsIkJDIl0sWzAsMCwiVEQiXSxbMCwzLCJGdW5jIl0sWzMsMCwiQ29tcCJdLFszLDEsIkRpbWVuc2lvbiJdLFszLDIsIkxhYmVsIl0sWzQsMiwiYmN2Il0sWzUsMiwiaW50ZWciLDAseyJvZmZzZXQiOi0xfV0sWzAsMiwib3YiLDFdLFswLDMsIm90IiwxXSxbMSwyLCJpdiIsMV0sWzEsMywiaXQiLDFdLFs1LDIsImRlcml2IiwyLHsib2Zmc2V0IjoxfV0sWzQsNiwiYmNmdW5jIiwyLHsic3R5bGUiOnsiYm9keSI6eyJuYW1lIjoiZGFzaGVkIn19fV0sWzMsNiwidGZ1bmMiLDAseyJzdHlsZSI6eyJib2R5Ijp7Im5hbWUiOiJkYXNoZWQifX19XSxbMiw5LCJzeW1ib2wiLDEseyJzdHlsZSI6eyJib2R5Ijp7Im5hbWUiOiJkYXNoZWQifX19XSxbMiw4LCJkaW1lbnNpb24iLDEseyJzdHlsZSI6eyJib2R5Ijp7Im5hbWUiOiJkYXNoZWQifX19XSxbMiw3LCJjb21wbGV4IiwxLHsic3R5bGUiOnsiYm9keSI6eyJuYW1lIjoiZGFzaGVkIn19fV1d
  @present TheoryTontiDiagram(FreeSchema) begin
    Func::Data
    Comp::Data
    Dimension::Data
    Label::Data

    V::Ob
    I::Ob
    O::Ob
    T::Ob
    BC::Ob
    TD::Ob

    iv::Hom(I,V)
    it::Hom(I,T)
    ov::Hom(O,V)
    ot::Hom(O,T)
    bcv::Hom(BC,V)
    deriv::Hom(TD,V)
    integ::Hom(TD,V)

    tfunc::Attr(T,Func)
    bcfunc::Attr(BC,Func)

    complex::Attr(V,Comp)
    dimension::Attr(V,Dimension)
    symbol::Attr(V, Label)
  end

  # Structure to "cache" these values. Ideally these would be stored in the
  # Petri net simulation layer, and initialized when the simulation layer
  # is constructed
  struct Space
    s::EmbeddedDeltaSet2D
    sd::EmbeddedDeltaDualComplex2D
    boundary
    hodge
    laplacian
    lie
  end

  function Space(s::EmbeddedDeltaSet2D{O, P}) where {O, P}
    sd = EmbeddedDeltaDualComplex2D{O, eltype(P), P}(s)
    subdivide_duals!(sd, Barycenter())

    boundary = Array{typeof(d(0,s)), 2}(undef, 2,2)
    boundary[[1,2,3,4]] .= [d(0,s), dual_derivative(0,sd), d(1,s), dual_derivative(1,sd)]

    hodge = Array{typeof(⋆(0,sd)), 2}(undef, 2,3)
    hodge[[1,2,3,4,5,6]] .= [⋆(0,sd), inv(⋆(2,sd)), ⋆(1,sd), -1 .* inv(⋆(1,sd)), ⋆(2,sd), inv(⋆(0,sd))]

    # TODO:
    # Add Laplacian and Lie operators here
    laplacian  = Array{typeof(hodge[1,2]*boundary[1,1]),1}(undef,3)
    laplacian[1] = hodge[2,3]*boundary[2,2]*hodge[1,2]*boundary[1,1]

    laplacian[2] = hodge[2,2]*boundary[2,1]*hodge[1,3]*boundary[1,2] .+
                   boundary[1,1]*hodge[2,3]*boundary[2,2]*hodge[1,2]
    laplacian[3] = boundary[1,2]*hodge[2,2]*boundary[2,1]*hodge[1,3]

    lie_op = [hodge[2,3]*boundary[2,2]*hodge[1,2], hodge[2,2]*boundary[2,1]*hodge[1,3]]
    lie = [(∂s,s,v)->(∂s .= boundary[1,2]*∧(Tuple{0,1},sd,s,bound[2,1]*v)),
           (∂u,u,v)->(∂u .= (star_1*∧(Tuple{1,0},sd,v,inv_star_0*cobound_1_2*u) .+ cobound_0_1*star_2*∧(Tuple{1,1},sd,v,inv_star_1*u))),

          ]
    Space(s, sd, boundary, hodge, laplacian, nothing)
  end

  # TODO:
  # Make more petri-style simulation ACSet
  #
  # Make more Tonti-aware ACSet w/ support of multiple complexes

  # These functions are necessary for defining a TontiDiagram data structure from
  # the ACSet
  const AbstractTontiDiagram = AbstractACSetType(TheoryTontiDiagram)
  const TontiDiagram = ACSetType(TheoryTontiDiagram,index=[:iv, :it, :ov, :ot, :bcv],
                                                   unique_index=[:symbol])

  # Define an interface for an OpenTontiDiagram which allows Tonti diagrams to
  # be composed over their corners
  const OpenTontiDiagramOb, OpenTontiDiagram = OpenACSetTypes(TontiDiagram,
                                                              :V)
  #==
  Open(td::AbstractTontiDiagram) = begin
    OpenTontiDiagram{Function, Symbol}(td,
                                       FinFunction(collect(1:nparts(td, :Corner)),
                                                   nparts(td, :Corner)))
  end==#

  TontiDiagram() = TontiDiagram{Function, Bool, Int64, Symbol}()

  function vectorfield(td, sp::Space)
    v_mem, t_mem = init_mem(td, sp)
    dg = Graph(td)

    order = topological_sort(dg)

    state_vars = filter(x->length(incident(td,x,:ov))==0, 1:nparts(td, :V))

    input_vars = Dict{Symbol, Tuple{Int64,Int64}}()
    cur_head = 1
    for i in state_vars
      v_size = length(v_mem[i])
      input_vars[td[i,:symbol]] = (cur_head,cur_head+v_size-1)
      cur_head += v_size
    end

    function system(du, u, t, p)
      for cur in order
        # Check if current is a variable or transition
        if cur > nparts(td, :V)
          cur -= nparts(td, :V)
          inputs = td[incident(td, cur, :it), :iv]
          outputs = incident(td, cur, :ot)
          td[cur,:tfunc](t_mem[outputs]..., v_mem[inputs]...)
        else
          inputs = incident(td, cur, :ov)
          if(length(inputs) == 0)
            # This means this is a state variable
            data_source = input_vars[td[cur,:symbol]]
            v_mem[cur] .= u[data_source[1]:data_source[2]]
          else
            v_mem[cur] .= 0
            for i in inputs
              v_mem[cur] .+= t_mem[i]
            end
          end
          bcs = incident(td, cur, :bcv)
          for bc in bcs
            td[bc, :bcfunc](v_mem[cur])
          end
        end
      end
      # If a state variable does not have a derivative defined, we keep it out
      # (we'll want to move these to the parameter argument instead)
      du .= 0
      for i in state_vars
        state_range = input_vars[td[i,:symbol]]
        out_var = td[incident(td, i, :integ), :deriv]
        if length(out_var) != 0
          du[state_range[1]:state_range[2]] .= v_mem[out_var[1]]
        end
      end
    end
    input_vars, system
  end

  function add_variable!(td, symbol::Symbol, dimension::Int64, complex::Bool)
    add_part!(td, :V,symbol=symbol, dimension=dimension, complex=complex)
  end

  function add_variables!(td, vars::Tuple{Symbol, Int64, Bool}...)
    for v in vars
      add_variable!(td, v...)
    end
  end

  function add_transition!(td, dom_sym::Array{Symbol,1}, func, codom_sym::Array{Symbol,1})
    dom = [findfirst(v->v == s, td[:symbol]) for s in dom_sym]
    codom = [findfirst(v->v == s, td[:symbol]) for s in codom_sym]
    t = add_part!(td, :T, tfunc=func)
    add_parts!(td, :I, length(dom), iv=dom, it=t)
    add_parts!(td, :O, length(codom), ov=codom, ot=t)
  end

  function add_derivative!(td, sp, dom_sym, codom_sym)
    dom = findfirst(v->v==dom_sym, td[:symbol])
    codom = findfirst(v->v==codom_sym, td[:symbol])

    # TODO:
    # Add tests for proper dimensions, complexes, etc.
    # This will later be replaced as we pre-initialize all boundary operators
    bound = sp.boundary[(td[dom,:complex] ? 1 : 2), td[dom,:dimension]+1]
    func(x,y) = (x.=bound*y)
    add_transition!(td, [dom_sym],func,[codom_sym])
  end

  function add_derivatives!(td, sp, vars::Pair{Symbol, Symbol}...)
    for v in vars
      add_derivative!(td, sp, v[1],v[2])
    end
  end

  function add_time_dep!(td, deriv_sym::Symbol, integ_sym::Symbol)
    deriv = findfirst(v->v==deriv_sym, td[:symbol])
    integ = findfirst(v->v==integ_sym, td[:symbol])

    add_part!(td, :TD, integ=integ, deriv=deriv)
  end

  function add_bc!(td, var_sym, func)
    var = findfirst(v->v==var_sym, td[:symbol])
    add_part!(td, :BC, bcfunc=func, bcv=var)
  end

  # Note: This function can be made more efficient if combined with existing
  # transformations.
  # e.g. Advection-diffusion can be merged after the initial wedge
  # product/coboundary operator
  #
  # Currently only defined on primal complices (can this be applied to dual
  # complices?)
  function add_laplacian!(td, sp, dom_sym, codom_sym; coef=1)
    sd = sp.sd
    dom = findfirst(v->v==dom_sym, td[:symbol])
    codom = findfirst(v->v==codom_sym, td[:symbol])

    lap_op = sp.laplacian[td[dom,:dimension]+1] # laplace_beltrami(Val{td[dom,:dimension]},sd)
    func(x,y) = (x .= coef * (lap_op*y))
    add_transition!(td, [dom_sym], func, [codom_sym])
  end

  function init_mem(td, s::EmbeddedDeltaSet1D)
    # Fill out this function
  end

  function init_mem(td, sp::Space)
    s = sp.s
    primal_size = [nv(s), ne(s), ntriangles(s)]
    dual_size   = [ntriangles(s), ne(s), nv(s)]

    t_mem = Array{Array{Float64,1},1}()
    v_mem = Array{Array{Float64,1},1}()

    for i in 1:nparts(td, :O)
      var = td[i,:ov]
      push!(t_mem, zeros(Float64,
                        td[var,:complex] ? primal_size[td[var,:dimension]+1] :
                        dual_size[td[var,:dimension]+1]))
    end

    for v in 1:nparts(td,:V)
      push!(v_mem, zeros(Float64,
                        td[v,:complex] ? primal_size[td[v,:dimension]+1] :
                        dual_size[td[v,:dimension]+1]))
    end
    v_mem, t_mem
  end

  function Graph(td)
    g = Graph()
    add_vertices!(g, nparts(td, :V) + nparts(td, :T))
    nvars = nparts(td, :V)
    for i in 1:nparts(td, :I)
      add_edge!(g, td[i,:iv], td[i,:it] + nvars)
    end
    for o in 1:nparts(td, :O)
      add_edge!(g, td[o,:ot] + nvars, td[o,:ov])
    end
    g
  end
end
