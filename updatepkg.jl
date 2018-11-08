# run this script before doing ]instantiate
# this works around a Pkg bug in 1.0.0
depot = first(Base.DEPOT_PATH)    
path = joinpath(depot, "clones")
for clone in readdir(path)
  @info "Updating $(joinpath(path, clone))"
  cd(joinpath(path, clone)) do
	  run(`git fetch`)
  end
end
