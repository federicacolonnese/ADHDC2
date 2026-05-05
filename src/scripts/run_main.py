from src.config import Config
from src.dataset.dataloaders_HDC import import_data, import_data_npy
from src.train.train_hdc import train_hdc, evaluate_hdc, train_hdc_graph, evaluate_hdc_graph

# Parametri
config = Config()
config.visualize = False

# Get data
if config.use_custom_npy:
    train_loader, test_loader, train_ds, test_ds, config = import_data_npy(
        config,
        x_path=config.npy_x_path,
        y_path=config.npy_y_path,
        batch_size=config.batch_size,
        test_size=config.npy_test_size,
    )
else:
    train_loader, test_loader, train_ds, test_ds, config = import_data(config,
                                                                       config.dataset_name,
                                                                       config.dataset_path,
                                                                       config.batch_size)

# Addestramento del modello HDC
if config.graph_bundling:
    prototipi = train_hdc_graph(config,
                                train_loader,
                                test_loader,
                                train_ds,
                                test_ds)
    metrics = evaluate_hdc_graph(test_loader, prototipi, config.device)
else:
    prototipi = train_hdc(config,
                          train_loader,
                          test_loader,
                          train_ds,
                          test_ds)
    # Evaluate model
    metrics = evaluate_hdc(test_loader, prototipi, config.device)
