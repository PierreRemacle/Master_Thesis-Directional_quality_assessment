import cProfile
import pstats
from Quality_all import ALL_path_2    # Import the function you want to profile

# Run your function with profiling enabled
cProfile.run('ALL_path_2()', 'profile_stats')

# Analyze the profiling data
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')  # Sort by cumulative time
stats.print_stats()
