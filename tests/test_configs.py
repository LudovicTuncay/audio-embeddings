import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_train_config_with_mock_data(cfg_train: DictConfig) -> None:
    """The train config should compose and instantiate with mock data overrides."""
    with open_dict(cfg_train):
        cfg_train.data = OmegaConf.create(
            {
                "_target_": "src.data.mock_audioset_datamodule.MockAudioSetDataModule",
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "max_audio_length_sec": 0.1,
                "target_sample_rate": 16000,
                "collate_mode": "pad",
            }
        )

    HydraConfig().set_config(cfg_train)

    datamodule = hydra.utils.instantiate(cfg_train.data)
    model = hydra.utils.instantiate(cfg_train.model)
    trainer = hydra.utils.instantiate(cfg_train.trainer)

    assert datamodule is not None
    assert model is not None
    assert trainer is not None
