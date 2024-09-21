import subprocess

for dataset in """
./data/lastfm_asia/lastfm_asia_edges.csv
./data/lastfm_asia/lastfm_asia_target.csv
""".splitlines():
    if dataset == "": continue

    with open(dataset, 'r') as file:
        for line in file.readlines():
            try:
                print(int(line))
            except ValueError as e:
                