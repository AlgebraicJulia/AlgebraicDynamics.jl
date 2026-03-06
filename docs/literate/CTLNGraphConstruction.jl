# # Constructing Valid Graphs for CTLN Fixed Point Computation
#
# This tutorial explains how to construct **valid CTLN graphs** and
# **simply embedded covers**.
#
# The fixed point support functor `\widehat{FP}` is defined only on objects of
# the category `Grph_se`, whose objects are **directed, loopless graphs** and
# whose morphisms are **simple embeddings**.
#
# In this framework, a *simply embedded cover* of a graph `G` is a collection of
# subgraphs simply embedded into `G` and whose vertex sets jointly cover the
# entire vertex set of `G`.
#
# This package supports two layers of construction:
# 1. building **cover elements**, and
# 2. assembling them into **simply embedded covers** via architecture constructors.
#
# ---
# ## 1. Supported Cover Elements
#
# A *cover element* may be **any directed, loopless Julia graph**.
# Users can construct such graphs in two ways.
#
# Begin by loading the package:
using AlgebraicDynamics
using AlgebraicDynamics.ThresholdLinear
using AlgebraicDynamics.ThresholdLinear: disjoint_union, cyclic_union, connected_union
using Catlab, Catlab.Graphs
using Catlab.Graphics

draw(g; kw...) = to_graphviz(g; node_labels=true, edge_labels=true, kw...)

#
# ### 1.1 Implicit graph constructors
#
# The file `graph_utils.jl` provides several *implicit graph types* that serve
# as building blocks.  
# These objects can be converted into concrete Catlab graphs using `Graph(element)`.

# A 2-clique (complete graph on 2 vertices):
C2  = CompleteGraph(2)
GC2 = Graph(C2)
draw(GC2)

# A directed 4-cycle:
Cy4  = CycleGraph(4)
GCy4 = Graph(Cy4)
draw(GCy4)

# A 5-vertex discrete graph (no edges):
D5  = DiscreteGraph(5)
GD5 = Graph(D5)
draw(GD5)

# The resulting graphs are directed, loopless, and have vertices numbered
# consecutively from `1` to `n`. These may now be used as cover elements in the
# architecture constructors introduced in the next section.
#
# ### 1.2 User-defined cover elements
#
# Any user-defined directed, loopless Julia graph (e.g., constructed with Catlab)
# is also a valid cover element.
#
# ---
# ## 2. Supported Simply Embedded Covers 
#
# Once we have cover elements (directed, loopless graphs), we can assemble them
# into larger architectures. The file `graph_utils.jl` provides several
# **architecture constructors** that glue cover elements together:
#
# - `disjoint_union`
# - `clique_union`
# - `cyclic_union`
# - `connected_union`
#
# Each of these takes two graphs and returns a new global graph `G`.  
# Conceptually, the original graphs together with their canonical embeddings
# into `G` form a *simply embedded cover*.
#
# Below we illustrate each constructor.
#
# ### 2.1 Disjoint unions
#
# `disjoint_union` places graphs side by side on disjoint blocks of vertices,
# without adding edges between components.

G_dis = disjoint_union(GC2, GCy4)
draw(G_dis)

# Here, the vertices of `GCy4` are relabeled to avoid overlap with `GC2`,
# and each original component embeds simply into `G_dis`. The family
# `{GC2, GCy4}` together with these embeddings forms a simply embedded cover
# of `G_dis`.
#
# ### 2.2 Clique unions
#
# `clique_union` starts from the disjoint union and then adds **all possible
# edges between the two components in both directions**, creating a fully
# bidirectionally connected bipartite structure between them.

G_clique = clique_union(GC2, GCy4)
draw(G_clique)

# As with `disjoint_union`, each component graph embeds simply into `G_clique`;
# the difference is that we now have additional edges between components.

# ### 2.3 Connected unions
#
# `connected_union` overlays two graphs that live on (possibly overlapping)
# vertex sets by **taking the union of their edge sets**. Recall:

draw(GC2)

#

draw(GCy4)

# These graphs share the subgraph `(1) --> (2)`, so their connected union is

G_conn = connected_union(GC2, GCy4)
draw(G_conn)

# In this case, the vertex labels are shared: both `GC2` and `GCy4` are
# viewed as graphs on (parts of) the same vertex set, and `connected_union`
# combines their edges into a single graph. The embeddings of
# `GC2` and `GCy4` into `G_conn` yield a simply embedded cover of `G_conn`.
#
# In all these examples, the resulting global graph `G` can be paired with the
# original components (and their embeddings) to obtain a simply embedded cover,
# which can then be used as input for CTLN fixed point computations.

# ---

# ## 3. Using These Graphs for Fixed Point Support Computations
#
# Each `(G, cover)` pair constructed above lies in the domain of the functor
# `\widehat{FP}`, so fixed point supports can be computed directly.

# ### 3.1 Binary covers

# Computing the fixed point supports of G_clique:
    
FPG_clique = FP(G_clique)

# and this is the distributed version
FPG_clique = FP(FP(GC2) + FP(GCy4))

# ---

# Disjoint, clique, and cyclic unions can be constructed with infix operators `+`, `*`, and `↻` (`\circlearrowright`).

fpg1 = erdos_renyi(Graph, 7, 0.3) |> FP
fpg2 = CycleGraph(100) |> FP
fpg12 = FP(fpg1 + fpg2)

# Here it is for a more complicated expression,

fpg3 = CompleteGraph(3) |> FP
fpg = FP(fpg3 ↻ (fpg1 + (fpg1 * fpg2)))


