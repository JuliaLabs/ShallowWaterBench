using Profile
using StatProfilerHTML

@time main()

@profile main()
@profile main()

statprofilehtml()
