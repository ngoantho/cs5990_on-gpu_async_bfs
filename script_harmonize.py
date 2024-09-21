from script_common import run

group_count = 240
cycle_count = 10000
run('./out/results_harmonize_event.csv', './harmonize.exe', 'event', '-group-count', str(group_count), "-cycle-count", str(cycle_count))
run('./out/results_harmonize_async.csv', './harmonize.exe', 'async', '-group-count', str(group_count), "-cycle-count", str(cycle_count))