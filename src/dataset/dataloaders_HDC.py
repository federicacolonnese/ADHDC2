import os

import numpy as np

import torchhd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Optional, Callable, Tuple
from aeon.datasets import load_classification

from src.dataset.hdc_encoding import HD_data_encoding
from src.config import Config
from src.dataset.features_extraction import TimeSeriesFeatureExtractor, FeaturesHDDataEncoding
from src.utils import plot_similarity_matrix


def import_data_npy(config: Config,
                    x_path: str,
                    y_path: str,
                    batch_size: int = 32,
                    test_size: float = 0.2,
                    random_state: int = 42):
    """
    Loads X.npy (N, C, T) and y.npy (N,), splits into train/test, returns DataLoaders.
    """
    X = np.load(x_path).astype(np.float32)   # (N, C, T)
    Y = np.load(y_path)                        # (N,)

    print(f"Loaded X: {X.shape}, Y: {Y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=test_size,
        stratify=Y,
        random_state=random_state,
    )

    train_loader, test_loader, train_ds, test_ds, config = build_loaders(
        config, X_train, y_train, X_test, y_test, batch_size=batch_size
    )
    config.num_classes = train_ds.num_classes
    config.input_features = X.shape[1]   # C (number of channels)
    return train_loader, test_loader, train_ds, test_ds, config


def import_data(config: Config,
                dataset_name: str = "PEMS-SF",
                dataset_path: str = r'C:\Users\Grid\Desktop\PhD\HDCGNN\code\GNNxHDC\dataset\aeon\PEMS_SF',
                batch_size: int = 64):

    # Choose raw_dataset e.g. "PEMS-SF" or "GunPoint", "ItalyPowerDemand", "BasicMotions", "Sleep", ...
    X_train, y_train = load_classification(dataset_name,
                                           split="train",
                                           extract_path=dataset_path)
    X_test, y_test = load_classification(dataset_name,
                                         split="test",
                                         extract_path=dataset_path)

    # Dataloaders
    train_loader, test_loader, train_ds, test_ds, config = build_loaders(config,
                                                                         X_train,
                                                                         y_train,
                                                                         X_test,
                                                                         y_test,
                                                                         batch_size=batch_size)

    config.num_classes = train_ds.num_classes

    # Plot similarity matrix train e test loader
    if config.visualize:
        train_hv = [elem[0] for elem in train_loader]
        test_hv = [elem[0] for elem in test_loader]
        train_hv = torch.cat(train_hv, dim=0)
        test_hv = torch.cat(test_hv, dim=0)
        print('Visualizing similarity matrix train loader!')
        plot_similarity_matrix(train_hv)
        print('Visualizing similarity matrix test loader!')
        plot_similarity_matrix(test_hv)


    return train_loader, test_loader, train_ds, test_ds, config



def build_loaders(config: Config,
                  X_tr,
                  Y_tr,
                  X_te,
                  Y_te,
                  batch_size=32,
                  shuffle=True,
                  num_workers=0):

    # Train raw_dataset, make CiM and other data useful for test data
    print('Processing TRAIN dataset ...')
    tr_ds = HVTimeSeriesDataset(X_tr,
                                Y_tr,
                                config,
                                flag='train')
    print('Processing TEST dataset ...')
    te_ds = HVTimeSeriesDataset(X_te,
                                Y_te,
                                config,
                                flag='test',
                                encoder=tr_ds.encoder,
                                feature_encoder=tr_ds.feature_encoder,
                                enc_and_feat_enc_key=tr_ds.enc_and_feat_enc_key,)

    tr_loader = DataLoader(tr_ds,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers)
    te_loader = DataLoader(te_ds,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers)
    return tr_loader, te_loader, tr_ds, te_ds, config


class HVTimeSeriesDataset(Dataset):
    """
    Wraps aeon-style outputs into a PyTorch Dataset.
    Expects each sample to be either:
      - numpy array with shape (N, C, L)  OR
      - 1D numpy array (L,) for univariate -> converted to (1, L)
      - object-dtype array of samples (common in aeon) -> each sample is handled individually
    Returns tensor with shape (C, L) per __getitem__.


    Final saved raw_dataset: (o hai tutto salvato o rifai tutto)
    d = ['train_cim': {c_0:"cim_c_0", c_1:"cim_c_1",......},
        'train_hv_data': hv_tensor_tr.npy,
        'test_hv_data': hv_tensor_te.npy]
    """

    def __init__(self,
                 X: Any,
                 Y: Any,
                 config: Config,
                 flag: str = 'train',
                 encoder: HD_data_encoding = None,
                 feature_encoder: FeaturesHDDataEncoding = None,
                 enc_and_feat_enc_key = None,
                 transform: Optional[Callable] = None,
                 dtype: torch.dtype = torch.float32,):
        self.transform = transform
        self.config = config
        self.encoder = None
        self.feature_encoder = None
        self.enc_and_feat_enc_key = None
        processed_dataset_path = ""
        processed_dataset_path_0 = ""
        processed_dataset_path_1 = ""
        filename = ""
        filename_0 = ""
        filename_1 = ""

        # Check if raw_dataset preprocessed directory exist, otherwise create it
        os.makedirs(os.path.join(config.processed_dataset_repo, config.dataset_name), exist_ok=True)
        
        # Check if HV data wrt current raw_dataset and configuration has been already processed
        dataset_format = 'NH' if config.sample_format == "first_dim_h" else ['NCH', 'NPH']
        cfg_key = f"{config.vsa}_H{config.dim_hv}_L{config.num_levels}_cb{int(config.channel_bundling)}"
        if dataset_format == 'NH':
            filename = f"hv_dataset_{config.dataset_name}_{dataset_format}_{cfg_key}_{flag}.npy"
            processed_dataset_path = os.path.join(config.processed_dataset_repo,
                                                  config.dataset_name,
                                                  filename)
        elif dataset_format == 'NCH':
            filename_0 = f"hv_dataset_{config.dataset_name}_{dataset_format[0]}_{cfg_key}_{flag}.npy"
            filename_1 = f"hv_dataset_{config.dataset_name}_{dataset_format[1]}_{cfg_key}_{flag}.npy"
            processed_dataset_path_0 = os.path.join(config.processed_dataset_repo,
                                                  config.dataset_name,
                                                  filename_0)
            processed_dataset_path_1 = os.path.join(config.processed_dataset_repo,
                                                  config.dataset_name,
                                                  filename_1)
        else:
            raise NotImplementedError

        # Check already processed data if exist
        if os.path.exists(processed_dataset_path) or \
            (os.path.exists(processed_dataset_path_0) and os.path.exists(processed_dataset_path_1)):  # and os.path.exists(config.processed_dataset_path_te):
            config.hv_data_already_processed = True
            print(f'Dataset {config.dataset_name} with configuration {dataset_format} already processed.')
        else:
            config.hv_data_already_processed = False
            print(f'Dataset {config.dataset_name} with configuration {dataset_format} not already processed.')


        # Load already processed raw_dataset
        if self.config.hv_data_already_processed and self.config.load_hv_dataset:
            print('Loading preprocessed datasets')
            if dataset_format == 'NH':
                hv_dataset_nh = np.load(processed_dataset_path)
                print(processed_dataset_path)
                print(f"Loaded HV raw_dataset as hv_dataset_{config.dataset_name}_{dataset_format}_{flag}.npy")
            elif dataset_format == ['NCH', 'NPH']:
                hv_dataset_nch = np.load(processed_dataset_path)
                hv_dataset_nph = np.load(processed_dataset_path)
                print(f"Loaded HV datasets as hv_dataset_{config.dataset_name}_{dataset_format[0]}.npy and "
                      f"hv_dataset_{config.dataset_name}_{dataset_format[1]}_{flag}.npy")
            else:
                raise ValueError(f"Dataset format {dataset_format} not supported")
            self.x_dataset = hv_dataset_nh  # TODO: adattare codice a casi nch e nph

        # Otherwise create processed raw_dataset
        else:
            print(f'Making preprocessed datasets {config.dataset_name} with:')

            # 1) Load training set
            print(f"- {flag.upper()} data of shape: {X.shape}")  # (n_samples, n_channels, T)
            print(f"- {flag.upper()} labels of shape: {Y.shape}")

            print(f"Using num_levels = {config.num_levels}")

            # Create HV with train data
            if flag == 'train':
                self.encoder = HD_data_encoding(num_timesteps=X.shape[-1],
                                           num_levels=config.num_levels,
                                           dim_hv=config.dim_hv,
                                           vsa=config.vsa,
                                           channel_bundling=False)
                self.enc_and_feat_enc_key = torchhd.random(2,
                                                            config.dim_hv,
                                                            config.vsa)
                print('Computing train encoding .....')
            else:
                self.encoder = encoder
                self.enc_and_feat_enc_key = enc_and_feat_enc_key

            # Here I encode X to HDC tensors, merging also channels if channel bundling is true
            hv_dataset = self.encoder.encode_dataset(X,config.channel_bundling)
            hv_dataset = hv_dataset.squeeze()

            # 3) Feature extraction
            if config.feature_extraction:
                # Get the features dataset
                dataset_features, list_of_pos_classes = self.get_dataset_for_features(X, config)

                # Create HV with train data for features
                if flag == 'train':
                    self.feature_encoder = FeaturesHDDataEncoding(list_of_pos_classes=list_of_pos_classes,
                                                                  num_levels=config.num_levels,
                                                                  dim_hv=config.dim_hv,
                                                                  vsa=config.vsa,
                                                                  channel_bundling=False)
                    print('Computing train encoding!')
                else:
                    self.feature_encoder = feature_encoder

                # Encode real numbers into HVs
                features_hv_dataset = self.feature_encoder.encode_dataset(dataset_features, config.channel_bundling_features)
                features_hv_dataset = features_hv_dataset.squeeze()
                print("FEATURES HV DATASET SHAPE", features_hv_dataset.shape)
                print("HV dataset", hv_dataset.shape)

                # Merge time series and features HV dataset
                hv_dataset = self.merge_hv_with_features(hv_dataset, features_hv_dataset)

            # Visualize CIM
            if config.visualize:
                for elem in self.encoder.cim_per_channel:
                    plot_similarity_matrix(elem.level_hv)  # exploratory HV analysis

            print(f"{flag.upper()} HDC dataset generated with shape:", list(hv_dataset.shape))  # (n_samples, dim_hv)
            self.x_dataset = hv_dataset

            # Save HV raw_dataset depending on NH or (NCH,NPH) raw_dataset configuration
            if config.save_hv_dataset:
                self.save_hdc_dataset(config, processed_dataset_path, filename_0, filename_1, hv_dataset)


        # Labels
        y_arr = np.asarray(Y).ravel()
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(y_arr).astype(np.int64)
        self.num_classes = torch.tensor(self.y).unique().shape[0]


    def get_dataset_for_features(self,
                                 X,
                                 config):
        list_of_pos_classes = list()

        # Define feature extraction class
        extractor = TimeSeriesFeatureExtractor(config)

        # Extract features for each samples
        dataset_features = list()
        print("SHAPE DATASET", np.shape(X))
        for sample in X:
            channels = list()
            for channel in sample:
                if self.config.feature_extracted == 'n_sins':
                    _, feat, list_of_pos_classes = extractor.extract_features_n_sins(channel)
                elif self.config.feature_extracted == 'ema_fft':
                    _, feat, list_of_pos_classes = extractor.extract_features_ema_fft(channel)
                else:
                    raise ValueError(f"Feature extracted {config.feature_extracted} not supported")
                channels.append(torch.tensor(feat))
            stacked_channels = torch.stack(channels, dim=0)

            dataset_features.append(stacked_channels)
        dataset_features = torch.stack(dataset_features, dim=0)  # List(C,T,F) -> (N,C,T,F)
        print("DATASET FEATURES SHAPE", dataset_features.shape)
        return dataset_features, list_of_pos_classes


    def merge_hv_with_features(self,
                               hv_dataset: torch.Tensor,
                               features_hv_dataset: torch.Tensor) -> torch.Tensor:
        """
        Combining temporal and features HVs

        :param hv_dataset: torch.tensor
        :param features_hv_dataset:
        :return:
        """
        # Check if temporal and features shapes are correspondent
        if hv_dataset.shape != features_hv_dataset.shape:
            raise ValueError(
                f"HV dataset shapes must match for bundling, got {hv_dataset.shape} and {features_hv_dataset.shape}")

        # Bind temporal and features to the corresponding key HVs
        hv_dataset_after_binding = torchhd.bind(hv_dataset,
                                                self.enc_and_feat_enc_key[0])
        hv_dataset_feature_after_binding = torchhd.bind(features_hv_dataset,
                                                        self.enc_and_feat_enc_key[1])

        # Combine HV data either by bundling or concatenating
        if self.config.temporal_and_feature_channels_bundling:
            merged_votes = torch.where(hv_dataset_after_binding + hv_dataset_feature_after_binding >= 0, 1.0, -1.0)
        else:
            merged_votes = torch.cat([hv_dataset_after_binding, hv_dataset_feature_after_binding], dim=1)

        return merged_votes

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if isinstance(self.x_dataset[idx], np.ndarray):
            x_t = torch.from_numpy(self.x_dataset[idx])  # dtype float32
        else:
            x_t = self.x_dataset[idx]
        if self.transform:
            x_t = self.transform(x_t)
        return x_t, int(self.y[idx])

    def get_label_encoder(self) -> LabelEncoder:
        return self.le

    def save_hdc_dataset(self, config, processed_dataset_path, filename_0, filename_1, hv_dataset):
        dataset_format = 'NH' if config.sample_format == "first_dim_h" else ['NCH', 'NPH']
        if dataset_format == 'NH':
            np.save(processed_dataset_path, hv_dataset.numpy())
            print(f"Saved HV raw_dataset as {processed_dataset_path}")
        elif dataset_format == ['NCH', 'NPH']:
            np.save(filename_0, hv_dataset.numpy())
            np.save(filename_1, hv_dataset.numpy())
            print(f"Saved HV datasets as {filename_0} and {filename_1}")
        else:
            raise ValueError(f"Dataset format {dataset_format} not supported")




