from script_common import run

run('./out/results_queue_directed.csv', './queue.exe', program=None, directed=True)
run('./out/results_queue_undirected.csv', './queue.exe', program=None, directed=False)