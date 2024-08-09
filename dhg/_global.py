from pathlib import Path


def get_dhg_cache_root():

    root = Path.home() / Path(".dhg/")
    root.mkdir(parents=True, exist_ok=True)
    return root


import sys,os
sys.path.append(str(os.getcwd()).split('Hyperedge')[0]+'Hyperedge')
from hedge import CACHE_ROOT,LOGS_ROOT


AUTHOR_EMAIL = "evanfeng97@gmail.com"
# global paths
# CACHE_ROOT = get_dhg_cache_root()




# DATASETS_ROOT = os.path.join(CACHE_ROOT, "dgh_datasets")

CACHE_ROOT = Path(CACHE_ROOT)

DATASETS_ROOT =  CACHE_ROOT / "dgh_datasets"
# REMOTE_ROOT = "https://data.deephypergraph.com/"
REMOTE_ROOT = "https://download.moon-lab.tech:28501/"
REMOTE_DATASETS_ROOT = REMOTE_ROOT + "datasets/"
# REMOTE_DATASETS_ROOT = "https://data.shrec22.moon-lab.tech:18443/DHG/datasets/"
