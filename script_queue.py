from script_common import run, get_root_node, datasets

def main_queue(dataset:str, root_node:int):
    run(
        "./out/results_queue_directed.csv",
        "./queue.exe",
        dataset,
        root_node,
        program=None,
        directed=True,
    )
    run(
        "./out/results_queue_undirected.csv",
        "./queue.exe",
        dataset,
        root_node,
        program=None,
        directed=False,
    )
