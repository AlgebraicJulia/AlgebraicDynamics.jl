using BenchmarkTools

const SUITE = BenchmarkGroup()

# Add some child groups to our benchmark suite. The most relevant BenchmarkGroup constructor
# for this case is BenchmarkGroup(tags::Vector). These tags are useful for
# filtering benchmarks by topic, which we'll cover in a later section.

include("diffusion_diffeq.jl")
include("diffusion_dyn.jl")
