using Documenter
using Literate

@info "Loading AlgebraicDynamics"
using AlgebraicDynamics
using AlgebraicDynamics.UWDDynam
using AlgebraicDynamics.DWDDynam
using AlgebraicDynamics.CPortGraphDynam
using Catlab
using Catlab.WiringDiagrams
using AlgebraicPetri
using DelayDiffEq
using OrdinaryDiffEq

# This is to resolve method ambiguity and is inlcuded in Base as of 1.8
if VERSION < v"1.8.0"
    Base.Sort.defalg(v::AbstractArray{Missing}) = Base.DEFAULT_UNSTABLE
    Base.Sort.defalg(v::AbstractArray{Union{}}) = Base.DEFAULT_UNSTABLE
end


# Set Literate.jl config if not being compiled on recognized service.
config = Dict{String,String}()
if !(haskey(ENV, "GITHUB_ACTIONS") || haskey(ENV, "GITLAB_CI"))
  config["nbviewer_root_url"] = "https://nbviewer.jupyter.org/github/AlgebraicJulia/AlgebraicDynamics.jl/blob/gh-pages/dev"
  config["repo_root_url"] = "https://github.com/AlgebraicJulia/AlgebraicDynamics.jl/blob/master/docs"
end

const literate_dir = joinpath(@__DIR__, "..", "examples")
const generated_dir = joinpath(@__DIR__, "src", "examples")

for (root, dirs, files) in walkdir(literate_dir)
  out_dir = joinpath(generated_dir, relpath(root, literate_dir))
  for file in files
    f,l = splitext(file)
    if l == ".jl" && !startswith(f, "_")
      Literate.markdown(joinpath(root, file), out_dir;
        config=config, documenter=true, credit=false)
      Literate.notebook(joinpath(root, file), out_dir;
        execute=false, documenter=true, credit=false)
    end
  end
end

extensions = ["AlgebraicPetri", "DelayDiffEq", "OrdinaryDiffEq"]
extension_modules = [Base.get_extension(AlgebraicDynamics, Symbol("AlgebraicDynamics"*ext*"Ext")) for ext in extensions]

@info "Building Documenter.jl docs"
makedocs(
  modules   = vcat([AlgebraicDynamics], extension_modules),
  format    = Documenter.HTML(
    assets = ["assets/analytics.js"],
  ),
  sitename  = "AlgebraicDynamics.jl",
  doctest   = false,
  checkdocs = :none,
  pagesonly = true,
  draft = false,
  pages     = Any[
    "AlgebraicDynamics.jl" => "index.md",
    "Examples" => Any[
      # "examples/induced_subgraphs.md",
      "examples/Lotka-Volterra.md",
      "examples/Ecosystem.md",
      "examples/CPG_examples.md",
      "examples/Cyber-Physical.md",
      "examples/Ross-Macdonald.md"
    ],
    "Threshold Linear Networks" => "TLN.md",
    "AlgebraicPetri Integration" => "AlgebraicPetri.md",
    "Library Reference" => "api.md"
  ]
)

@info "Deploying docs"
deploydocs(
  target = "build",
  repo   = "github.com/AlgebraicJulia/AlgebraicDynamics.jl.git",
  branch = "gh-pages"
)
