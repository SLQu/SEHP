from pathlib import Path
import os

# def get_dhg_cache_root():

#     root = Path.home() / Path(".dhg/")
#     root.mkdir(parents=True, exist_ok=True)
#     return root

def get_cache_root(project_name=''):
    import os
    root = f"{os.path.abspath(__file__).split(project_name)[0]}{project_name}/data"
    if not os.path.exists(root):
        os.mkdir(root)
    return root


# global paths
CACHE_ROOT = get_cache_root(project_name = 'SEHP')
# DATASETS_ROOT = CACHE_ROOT / "datasets"
LOGS_ROOT = f"{CACHE_ROOT}/logs"
if not os.path.exists(LOGS_ROOT):
    os.mkdir(LOGS_ROOT)
# LANGUAGE_MODEL_CACHE_PATH = os.path.join(CACHE_ROOT, 'language_models')


