import cProfile
profiler = cProfile.Profile()
profiler.enable()
# Run your engine here
profiler.disable()
profiler.print_stats(sort='time')
