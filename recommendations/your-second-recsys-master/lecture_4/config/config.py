import yaml
from pathlib import Path

config_path = Path(__file__).parent.joinpath("config.yaml").resolve()
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

recommender_conf = config["recommender"]
path_conf = config["paths"]