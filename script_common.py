import subprocess
import random
from os import path

datasets = """
./data/comm-f2f-Resistance/network_list.csv
./data/DIMACS/johnson8-2-4.mtx
./data/DIMACS/johnson8-2-4_nocomment.mtx
./data/DIMACS/MANN-a9.mtx
./data/DIMACS/san400-0-5-1.mtx
./data/DIMACS/brock200-1.mtx
./data/DIMACS/C500-9.mtx
./data/DIMACS/C1000-9.mtx
./data/DIMACS/C4000-5.mtx
./data/lastfm_asia/lastfm_asia_edges.csv
./data/lastfm_asia/lastfm_asia_target.csv
./data/facebook_clean_data/artist_edges.csv
./data/facebook_clean_data/athletes_edges.csv
./data/facebook_clean_data/company_edges.csv
./data/facebook_clean_data/government_edges.csv
./data/facebook_clean_data/new_sites_edges.csv
./data/facebook_clean_data/politician_edges.csv
./data/facebook_clean_data/public_figure_edges.csv
./data/facebook_clean_data/tvshow_edges.csv
"""

def subprocess_call(process:str, dataset:str, root_node:str, program:str|None, directed:bool, *args):
  return subprocess.run([process, '-file', dataset, '-root', root_node, "-program", program if program != None else str(None), "-directed" if directed else "", *args], stdout=subprocess.PIPE)

def run(output_filename:str, process:str, program:str, *args:list[str]):
  output = open(output_filename, "w")
  output.write("dataset,root node,program,directed,runtime (ms)\n")

  for dataset in datasets.splitlines():
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
    dataset_filename = path.basename(dataset)

    call_directed = subprocess_call(process, dataset, root_node, program, True, *args)
    runtime_directed = call_directed.stdout.decode().strip()
    output.write(f"{dataset_filename},{root_node},{program},true,{runtime_directed}\n")

    call_undirected = subprocess_call(process, dataset, root_node, program, False, *args)
    runtime_undirected = call_undirected.stdout.decode().strip()
    output.write(f"{dataset_filename},{root_node},{program},false,{runtime_undirected}\n")

  output.close()
