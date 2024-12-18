from script_common import run, get_root_node, datasets

group_count = 240
cycle_count = 10000


def main_harmonize(dataset: str, root_node: int):
    run(
        "./out/results_harmonize_event_directed.csv",
        "./src/harmonize.exe",
        dataset,
        root_node,
        "event",
        True,
        "-group-count",
        str(group_count),
        "-cycle-count",
        str(cycle_count),
    )
    run(
        "./out/results_harmonize_event_undirected.csv",
        "./src/harmonize.exe",
        dataset,
        root_node,
        "event",
        False,
        "-group-count",
        str(group_count),
        "-cycle-count",
        str(cycle_count),
    )

    run(
        "./out/results_harmonize_async_directed.csv",
        "./src/harmonize.exe",
        dataset,
        root_node,
        "async",
        True,
        "-group-count",
        str(group_count),
        "-cycle-count",
        str(cycle_count),
    )
    run(
        "./out/results_harmonize_async_undirected.csv",
        "./src/harmonize.exe",
        dataset,
        root_node,
        "async",
        False,
        "-group-count",
        str(group_count),
        "-cycle-count",
        str(cycle_count),
    )


if __name__ == "__main__":
    for dataset in datasets:
        root_node = get_root_node(dataset)
        main_harmonize(dataset, root_node)
