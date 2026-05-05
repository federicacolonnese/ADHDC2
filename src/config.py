import os
import torch
from pathlib import Path


class Config:
    def __init__(self):
        # Dataset parameters
        self.project_dir = Path(__file__).resolve().parents[1]
        self.dataset_name: str = "ieee_adhd"  # used as folder/filename key for HV cache; change when using aeon datasets
        self.dataset_path: str =  os.path.join(self.project_dir,
                                               'data',
                                               'raw_dataset',
                                               'aeon')
        self.processed_dataset_repo: str = os.path.join(self.project_dir,
                                                        'data',
                                                        'hv_processed_data')
        self.registry_path: str = os.path.join(self.project_dir,
                                                        'registry')
        os.makedirs(self.registry_path, exist_ok=True)

        # Custom .npy dataset (ADHD IEEE)
        self.use_custom_npy: bool = True
        self.npy_x_path: str = '/home/beingfedericax/Documenti/phd/ADHDC_old/ADHDC/dataset/ieee_dataset/x.npy'
        self.npy_y_path: str = '/home/beingfedericax/Documenti/phd/ADHDC_old/ADHDC/dataset/ieee_dataset/y.npy'
        self.npy_test_size: float = 0.2   # fraction of samples for test set

        self.save_hv_dataset: bool = True   # save encoded HV data to disk after first run
        self.load_hv_dataset: bool = True   # load from disk if cache exists
        self.hv_data_already_processed: bool = False
        self.sample_format: str = "first_dim_h"  # ["first_dim_h", "first_dim_c"]

        # Training parameters
        self.batch_size: int = 32
        self.shuffle : bool = True
        self.num_workers: int = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.visualize : bool = False
        self.epochs: int = 100


        # HDC parameters
        self.num_levels: int = 120
        self.n_values_feat = (2,4)
        self.alpha: float = 0.3
        self.k_fft: int = 3
        self.dim_hv: int = 10000
        self.max_iter: int = 1
        self.vsa = 'MAP'
        self.graph_bundling : bool = False  # <------------------------------------------- choose if use graph bundling
        self.feature_extraction : bool = False    # <-------------------------------------------------- activate feature
        self.channel_bundling : bool = True   # <-------------------------------- classical HDC bundling along channels
        self.channel_bundling_features : bool = True # <---------------- classical HDC bundling along features channels
        self.temporal_and_feature_channels_bundling : bool = False # <------ merge or cat temporal and features channels
        self.feature_extracted: str = 'n_sins'  # ('n_sins', 'ema_fft')

        self.input_features: int = 19  # C: number of EEG channels (overwritten at runtime)
        self.num_classes: int = 2      # ADHD vs control (overwritten at runtime)

    def __str__(self):
        lines = ["Config:"]
        for key, value in vars(self).items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)