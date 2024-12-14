from script_common import run, get_root_node, datasets


def main_queue(dataset: str, root_node: int):
    run(
        "./out/results_queue_directed.csv",
        "./src/queue.exe",
        dataset,
        root_node,
        program=None,
        directed=True,
    )
    run(
        "./out/results_queue_undirected.csv",
        "./src/queue.exe",
        dataset,
        root_node,
        program=None,
        directed=False,
    )


if __name__ == "__main__":
    for dataset in datasets:
        root_node = get_root_node(dataset)
        main_queue(dataset, root_node)