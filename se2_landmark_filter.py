import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf/", config_name="se2_landmark_filter.yaml")
def main(cfg: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from examples.se2_landmark_filter import main as se2_filter

    return se2_filter(cfg)


if __name__ == "__main__":
    main()
