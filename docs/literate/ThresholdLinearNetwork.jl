using AlgebraicDynamics

# Example of construction of a graph using motives.

C3 = GluingExpression(C(3))
D2 = GluingExpression(D(2))
D3 = GluingExpression(D(3))

#
vD2 = nv(collect(D2)) |> last
D2D2 = D2 + D2
D2D2_collected = collect(D2D2)
eD2D2 = edges(D2D2_collected) |> last
@assert eD2D2 == 2*vD2*vD2 

# Disjoint union of C3 and D3, only 3 edges from C3
G1 = C3 * D3

# Clique union of the previous graph G1 and another cycle C3, adding 3 edges and 6*2*3 edges between the two cliques
G2 = G1 + C3

# Clique union of the previous graph G2 and another cycle C3, adding 3 edges and 9*3*2 edges between the two cliques
G3 = G2 + C3

# Add another two C3 motives (3+3 nodes)
G4 = G3 + (C3*C3)

# Clique union of the previous graph G4 and a discrete graph D2, adding 2 edges and 21*2*3 edges between the two cliques
G5 = G4 + D2

# Build partition from motives in G3.
partition = vcat(fill(1, nv(C3)), fill(2, nv(D3)), fill(3, nv(C3)))
# @assert length(partition) == nv(G5)
cover = cover_partition(partition)
@assert cover == [[1,2,3], [4,5,6], [7,8,9]]

# Transform local supports to global indices
 
locals = [
    # piece 1 (C3), lifted by τ₁ = [1,2,3]
    Support.([[1,2,3]]),
    # piece 2 (D3), lifted by τ₂ = [4,5,6]
    Support.([[4], [5], [6], [4,5], [4,6], [5,6], [4,5,6]]),
    # piece 3 (C3), lifted by τ₃ = [7,8,9]
    Support.([[7,8,9]])
]

# # example with 3 blocks
local_supports = [
    # ex [[1,2,3]] -> [[1,2,3]]
    local_sup(Local(C3._1), cover[1]),  
    # ex [[1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]] -> [[4], [5], [6], [4,5], [4,6], [5,6], [4,5,6]]
    local_sup(Local(D3._1), cover[2]),
    # ex supp_C3 = [[1,2,3]] -> [[7,8,9]]
    local_sup(Local(C3._1), cover[3])
]

@assert locals == local_supports

partition = vcat(
    fill(1, nv(C3)),
    fill(2, nv(D3)),
    fill(3, nv(C3)),
    fill(4, nv(C3)),   
    fill(5, nv(C3)),
    fill(6, nv(C3)),
    fill(7, nv(D2)),
)
cover = cover_partition(partition)

function benchmark_local(graph, cover)
    supp_C3 = Local(C3._1)    # ex supp_C3 = [[1,2,3]]
    supp_D3 = Local(D3._1)    # ex supp_D3 = [[1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]
    supp_D2 = Local(D2._1)    # ex [[1],[2],[1,2]]
    locals = Local.([
        local_sup(supp_C3, cover[1]),  # C3
        local_sup(supp_D3, cover[2]),  # D3
        local_sup(supp_C3, cover[3]),  # C3
        local_sup(supp_C3, cover[4]),  # C3
        local_sup(supp_C3, cover[5]),  # C3
        local_sup(supp_C3, cover[6]),  # C3
        local_sup(supp_D2, cover[7])   # D2
       ])
    # ex locals = [[[1, 2, 3]], [[4], [5], [6], [4, 5], [4, 6], [5, 6], [4, 5, 6]], 
    # [[7, 8, 9]], [[10, 11, 12]], [[13, 14, 15]], [[16, 17, 18]], [[19], [20], [19, 20]]]
    # net = CTLNetwork(G2)
    net = CTLNetwork(graph)
    Supports(net, locals)
end

G5_collected = collect(G5)

@time fps_cover = benchmark_local(G5_collected, cover)

function benchmark_bruteforce(n)
    net = CTLNetwork(n)
    tln = TLNetwork(net)
    supports = Supports(tln)
    return supports
end

@time fps_bruteforce = benchmark_bruteforce(G5_collected)

# TODO sort
@test sort(fps_cover) == sort(fps_bruteforce)


# compute_all_local_supports(graph, cover)
# which is combining compute_local_supports and local_sup for each block
# store the graph G as a tree of (clique or disjoint) unions of subgraphs/motives,
# then recursively compute the local supports for each subgraph/motive and lift them to global indices
# finally combine the local supports to get the global supports
# then do local to global mapping for each local support
# once this is done, we can run the code below on an arbitrary graph with a fixed cover.

# Storing the graph as a tree of (clique or disjoint) unions of subgraphs/motives is not implemented yet,
# but will be a great way to store the graph and its cover because it doesn't need to realize the whole adjacency matrix of the graph.
# We can just store the motives and the way they are combined to get the graph.
# This will be useful for large graphs where the adjacency matrix is too large to store in memory.
# We need an evaluate function that takes the tree and returns the actual graph to run the brute force algorithm on the whole graph.

nblocks = maximum(partition)  # number of blocks in the cover
Knb = Catlab.Graphs.complete_graph(Graph, nblocks)  # complete graph on nblocks nodes
# add self-loops to Knb
for v in vertices(Knb)
    add_edge!(Knb, v, v)
end
phi, = homomorphisms(collect(G5), Knb, initial=(V=partition,))

to_graphviz(phi, node_attrs=Dict(:style=>"filled"), node_labels=true, node_colors=true)

# # just to visualize the graph

G1 = C3 + C3

G2 = G1 * Term(C2)

fps_G2 = Local(collect(G2))

G3 = G2 + Term(C2)

fps_G3 = Local(collect(G3))
