# Induced Subgraphs as a Sheaf
using Catlab
using Catlab.Graphs
using Catlab.Graphics
# using Catlab.Sheaves


draw_subobject = to_graphviz ∘ dom ∘ hom
draw(g) = to_graphviz(g, node_labels=true)
is_subobject(X::Subobject,Y::Subobject) = force(meet(X,Y)) == force(X)

function restrict(g::Graph, U::FinFunction)
  codom(U) == vertices(g) && error("U is not a subset of the vertex set:$U, $(vertices(g))")
  h₀ = Subobject(g, V=collect(U))
  h = negate(negate(h₀))
  return h
end

"""    is_cover(S::FinSet, opens::Multcospan)

Validates that the union of subsets is the entire set
This is used in the constructor for ModuleCover.
"""
function is_cover(S::FinSet, opens::Multicospan)
  sort(foldl(∪, map(opens) do Ui
    collect(Ui)
  end)) == collect(S)
end

"""    is_cover(G::Graph, opens::Multcospan)

Validates that the union of the induced subgraphs is the entire graph.
This is used in the constructor for ModuleCover.

TODO: There has to be a better way to check if a subgraph is the whole graph
"""
function is_cover(G::Graph, opens::Multicospan)
  g = foldl(join,
      map(legs(opens)) do Ui
          restrict(G, Ui)
      end
  )
  collect(g.components.V) == vertices(G) && collect(g.components.E) == 1:ne(G)
end

"""    ModuleCover(G::Graph, opens::Multicospan)

A struct for storing a graph along with a covering family of induced subgraphs.
We store the collect of vertex sets as a multicospan in FinSet. And this induces the
covering family on G. The constructor checks that every vertex and every edge is in at least one open.
"""
struct ModuleCover
  G::Graph
  opens::Multicospan
  function ModuleCover(G::Graph, opens::Multicospan)
    length(apex(opens)) == nv(G) || error("opens must cover the vertex set of the graph.")
    is_cover(apex(opens), opens) || error("Not every vertex in G appears in an induced subgraph.")
    is_cover(G, opens) || error("Not every edge in G appears in an induced subgraph formed by restriction along the opens.")
    return new(G, opens)
  end
end

Base.getindex(cov::ModuleCover, i) = legs(cov.opens)[i]

function Base.show(io::IO, cov::ModuleCover)
  println(io, "ModuleCover:")
  show(io, cov.G)
  print(io, "\nCovering Family:")
  map(enumerate(cov.opens)) do (i,Ui)
    print(io,"\n  U[$i]: ")
    show(io, collect(Ui))
  end
end

# ```julia
# struct CTLNCover
#   ctln::CTLNetwork
#   cov::ModuleCover
#   function CTLNCover(ctln::CTLNCover, cov::ModuleCover)
#     apex(cov) == ctln.G || error("CTLN and Cover don't have the same graph")
#     return new(ctln, cov)
#   end
# end
# ```

using Test

g = @acset Graph begin
  V = 3
  E = 3
  src = [1,2,3]
  tgt = [2,3,1]
end

@test_throws Exception ModuleCover(g,
              Multicospan(FinSet(3),
                [FinFunction([1], 3),
                FinFunction([2,1], 3),
                  FinFunction([2,3], 3)]))

# @test collect(cov[1]) == [1]
# @test collect(cov[2]) == [2,1]
# @test collect(cov[3]) == [2,3]
# restrict(g, cov[2])
# restrict(g, legs(cov.opens)[3])
# draw(restrict(g, cov[2]))
# draw(restrict(g, cov[3]))
# draw(join(restrict(g, cov[2]), restrict(g, cov[3])))
# draw(join(join(restrict(g, cov[1]), restrict(g, cov[2])), restrict(g, cov[3])))
# @test is_cover(cov) == false

cov2 = ModuleCover(g,
       Multicospan(FinSet(3),
        [FinFunction([1,3], 3),
         FinFunction([2,1], 3),
          FinFunction([2,3], 3)]))

@test is_cover(FinSet(3), cov2.opens)

# draw(restrict(cov2.G, cov2[1]))
# draw(restrict(cov2.G, cov2[2]))
# draw(restrict(cov2.G, cov2[3]))

