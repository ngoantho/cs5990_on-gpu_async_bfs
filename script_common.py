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
""".splitlines()


def subprocess_call(
    process: str,
    dataset: str,
    root_node: int,
    program: str,
    directed: bool,
    *args,
):
    return subprocess.run(
        [
            process,
            "-file",
            dataset,
            "-root",
            str(root_node),
            "-program",
            program if program != None else str(None),
            "-directed" if directed else "",
            *args,
        ],
        stdout=subprocess.PIPE,
    )


def get_root_node(dataset: str) -> int:
    nodes = []
    if dataset == "":
        return

    with open(dataset, "r") as file:
        delimiter = None
        if dataset.endswith("mtx"):
            delimiter = " "
        elif dataset.endswith("csv"):
            delimiter = ","
        else:
            raise Exception("unsupported file format")

        for line in file.readlines():
            splitted = line.split(delimiter)
            try:
                node, edge = int(splitted[0]), int(splitted[1])
                nodes.append(node)
            except Exception as e:
                pass
    return random.choice(nodes)


def run(
    output_filename: str,
    process: str,
    dataset: str,
    root_node: int,
    program: str,
    directed: bool,
    *args: list[str],
):
    with open(output_filename, "a") as output:
        if dataset == "":
            return
        elif "#" in dataset:
            return
        
        print(output_filename, process, dataset, '-root '+root_node, program, '-directed '+directed, *args)
        dataset_filename = path.basename(dataset)

        call = subprocess_call(process, dataset, root_node, program, directed, *args)
        runtime = call.stdout.decode().strip()
        output.write(f"{dataset_filename},{root_node},{program},{directed},{runtime}\n")
