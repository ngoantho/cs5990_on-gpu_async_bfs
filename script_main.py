from script_common import get_root_node, datasets
from script_queue import main_queue
from script_harmonize import main_harmonize

for dataset in datasets:
  root_node = get_root_node(dataset)

  main_queue(dataset, root_node)
  main_harmonize(dataset, root_node)