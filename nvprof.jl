import NVTX
import CUDAdrv

# NOTE: use with `--profile-from-start off`
NVTX.stop()

@info "First time + compilation"
@time main()

@info "Start profile"
NVTX.mark("main")
NVTX.@activate CUDAdrv.@profile begin
  main()
end
