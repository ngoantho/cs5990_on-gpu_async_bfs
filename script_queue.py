import subprocess
import random

for dataset in """
# ./data/DIMACS/johnson8-2-4.mtx
./data/DIMACS/MANN-a9.mtx
# ./data/lastfm_asia/lastfm_asia_edges.csv
# ./data/lastfm_asia/lastfm_asia_target.csv
""".splitlines():
    if dataset == "": continue
    elif "#" in dataset: continue

    nodes = []
    with open(dataset, 'r') as file:
        delimiter = None
        if dataset.endswith("mtx"): delimiter = " "
        elif dataset.endswith("csv"): delimiter = ","
        else: raise Exception("unsupported file format")

        for line in file.readlines():
            splitted = line.split(delimiter)
            try:
                node, edge = int(splitted[0]), int(splitted[1])
                nodes.append(node)
            except Exception as e:
                pass
    
    root_node = str(random.choice(nodes))
    call = subprocess.run(['./queue.exe', '-file', dataset, '-root', root_node], stdout=subprocess.PIPE)
    runtime = call.stdout.decode()

    print(runtime)