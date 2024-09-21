from script_common import run

group_count = 240
cycle_count = 10000

run('./out/results_harmonize_event_directed.csv', './harmonize.exe', 'event', True, '-group-count', str(group_count), "-cycle-count", str(cycle_count))
run('./out/results_harmonize_event_undirected.csv', './harmonize.exe', 'event', False, '-group-count', str(group_count), "-cycle-count", str(cycle_count))

run('./out/results_harmonize_async_directed.csv', './harmonize.exe', 'async', True, '-group-count', str(group_count), "-cycle-count", str(cycle_count))
run('./out/results_harmonize_async_undirected.csv', './harmonize.exe', 'async', False, '-group-count', str(group_count), "-cycle-count", str(cycle_count))