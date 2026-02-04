# Tutorial

In [last section], we've shown that certain graphs, when combined in a certain way, allow us to infer the fixed points of dynamical systems on these graphs. 

The basic graphs whose dynamics has predictable fixed points are the cycle graph, complete graph, and discrete graph. These graphs are all specified by an integer $n$ of vertices, which can be potentially quite large. Since we know how to infer fixed points from dynamics of specific combinations of graphs, it's not necessary to materialize a graph data structure in memory. Therefore these constructors are *implicit graphs*, or data structures which only store their vertices.

```@example
c4 = CycleGraph(4)
k3 = CompleteGraph(3)
d5 = DiscreteGraph(5)
```

Each implicit graph may become an *explicit graph* by passing it into Catlab's `Graph` constructor method.

```@example
@assert Graph(d5) == Graph(5)
@assert Graph(k3) == complete_graph(Graph, 3)
```

They may be combined through connected unions, comet

```@example
c3 = CycleGraph(3)
```

# Fixed Point Supports

In [the previous section] we defined the **fixed point support** as a set of vertices which the orbit of a fixed point lies. More precisely, the fixed point support functor `FP` is a separated presheaf (and sometimes a sheaf).

We have an API for constructing fixed point supports, assigning disjoint unions to addition `+`, clique unions to multiplication `*`, and cyclic unions to `↻` (\circleright) operators.

```@example
fpg1 = erdos_renyi(Graph, 7, 0.3) |> FP
fpg2 = CycleGraph(100) |> FP
fpg12 = FP(fpg1 + fpg2)
```

Here it is for a more complicated expression,

```@example
fpg3 = CompleteGraph(3) |> FP
fpg = FP(fpg3 ↻ (fpg1 + (fpg1 * fpg2)))
```
