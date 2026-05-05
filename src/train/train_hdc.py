from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.models.graph_hdc import GraphBundlerGlobal


def train_hdc(config,
              train_loader,
              test_loader,
              train_ds,
              test_ds):
    print('\nStart training ...')
    num_classes = config.num_classes
    dim_space = config.dim_hv
    max_iter = config.max_iter
    device = config.device

    # Prototipi inizializzati a zero (accumulatori)
    prototipi = torch.zeros((num_classes, dim_space), device=device)

    def update_prototypes(x, y):
        class_id = y.item()  # <-- Converte da tensor([k]) a scalare k
        prototipi[class_id] += x.squeeze()

    def binarize_prototypes():
        # HV bipolari: {-1, +1}
        # return torch.sign(prototipi)
        return torch.where(prototipi >= 0, torch.tensor(1.0), torch.tensor(-1.0))

    # TRAINING
    for iter in range(max_iter):
        print(f"Iterazione {iter+1}/{max_iter}")

        pbar = tqdm(train_loader, desc="Training", dynamic_ncols=True)

        # Adattare tipo le connesisoni tra i canali in funzioni dei dati in input e di una loss
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Aggiorno i prototipi sommando gli HV del batch
            for i in range(x_batch.size(0)):
                update_prototypes(x_batch[i], y_batch[i])
                # TODO: confrontare hypervettori della stessa classe e prendere i piu lontani

        if config.vsa == 'MAP':
            # Binarizzazione HDC PREFERITA
            prototipi = binarize_prototypes()

    # Evaluate model on training set
    evaluate_hdc(train_loader,
                 prototipi,
                 config.device,
                 flag='train')


    return prototipi




def evaluate_hdc(loader, prototipi, device, flag='test'):
    """
    Valuta il modello sui dati di test e calcola le metriche.

    Returns:
    - metrics: Dizionario con accuracy, precision, recall e f1-score
    """
    print(f'\nStart test on {flag} set...')
    correct = 0
    total = 0
    predictions = []
    labels = []

    prototipi = prototipi.to(device)

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            for i in range(x_batch.size(0)):
                x = x_batch[i]
                y_true = y_batch[i]

                # HV già calcolato
                high_dim_x = x

                # Similarità coseno
                similarities = cosine_similarity(high_dim_x.cpu().numpy().reshape(1, -1),
                                                 prototipi.cpu().numpy())
                y_pred = np.argmax(similarities)

                correct += (y_pred == y_true).item()
                total += 1

                predictions.append(y_pred)
                labels.append(y_true.item())

    # Metriche
    accuracy = correct / total
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    return metrics



def train_hdc_graph(config,
              train_loader,
              test_loader,
              train_ds,
              test_ds):
    """
    Alternating training:
      1) Build prototypes with current graph (no grad)
      2) Train ONLY the graph (backprop) with prototypes fixed
    """
    print('\nStart training ...')
    num_classes = config.num_classes
    dim_space = config.dim_hv
    max_iter = config.max_iter
    device = config.device

    channel_space = train_ds[0][0].shape[0]
    mp_steps = getattr(config, "mp_steps", 2)

    graph_step = GraphBundlerGlobal(
        C=channel_space,
        steps=mp_steps,
        symmetric=getattr(config, "symmetric_A", True),
        self_loops=getattr(config, "self_loops", True),
    ).to(device)

    # optimizer SOLO sul grafo
    lr_graph = getattr(config, "lr_graph", 1e-4)
    wd_graph = getattr(config, "wd_graph", 0.0)
    optimizer = torch.optim.Adam(graph_step.parameters(), lr=lr_graph, weight_decay=wd_graph)

    # quante epoche di backprop sul grafo per ogni outer-iter
    graph_epochs = getattr(config, "graph_epochs_per_outer", 5)

    # temperatura per CE su cosine logits
    temperature = getattr(config, "temperature", 0.5)
    # label_smooth = getattr(config, "label_smoothing", 0.1)

    # regolarizzazioni opzionali sull'adjacency
    reg_l1_A = getattr(config, "reg_l1_A", 0.1)          # sparsity
    reg_entropy_A = getattr(config, "reg_entropy_A", 0.1) # selettività (min entropia)

    def binarize_prototypes(prototipi):
        return torch.where(prototipi >= 0,
                           torch.tensor(1.0, device=prototipi.device),
                           torch.tensor(-1.0, device=prototipi.device))

    def cosine_logits_torch(hv_bh,
                            protos_kh,
                            eps=1e-8):
        # torch-only cosine logits: (B,K)
        hv_bh = F.normalize(hv_bh, dim=-1, eps=eps)
        protos_kh = F.normalize(protos_kh, dim=-1, eps=eps)
        return hv_bh @ protos_kh.t()

    @torch.no_grad()
    def build_prototypes():
        """Proto-step: prototipi Hebbian con grafo corrente (NO grad)."""
        graph_step.eval()
        prototipi = torch.zeros((num_classes, dim_space), device=device)

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)  # (B,C,H)
            y_batch = y_batch.to(device)  # (B,)

            bundle, _ = graph_step(x_batch)  # (B,H)
            bundle = binarize_prototypes(bundle)
            # Accumulo per classe
            for c in range(num_classes):
                idx = (y_batch == c)
                if idx.any():
                    prototipi[c] += bundle[idx].sum(dim=0)

        if getattr(config, "vsa", None) == 'MAP':
            prototipi = binarize_prototypes(prototipi)

        return prototipi

    def train_graph_one_epoch(prototipi_fixed):
        """Graph-step: backprop SOLO sul grafo con prototipi fissi."""
        graph_step.train()
        prototipi_fixed = prototipi_fixed.detach()

        total_loss = 0.0
        total = 0
        correct = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)  # (B,C,H)
            y_batch = y_batch.to(device)  # (B,)

            optimizer.zero_grad(set_to_none=True)

            bundle, A = graph_step(x_batch)  # bundle (B,H), A (C,C)

            bundle = torch.where(bundle >= 0,
                        torch.tensor(1.0, device=bundle.device),
                        torch.tensor(-1.0, device=bundle.device))

            logits = cosine_logits_torch(bundle, prototipi_fixed) / temperature  # (B,K)
            loss = F.cross_entropy(logits, y_batch)

            # reg opzionali su A
            if reg_l1_A > 0:
                loss = loss + reg_l1_A * A.abs().mean()

            if reg_entropy_A > 0:
                # entropia media delle righe (A è row-stochastic)
                ent = -(A * (A.clamp_min(1e-8)).log()).sum(dim=-1).mean()
                loss = loss + reg_entropy_A * ent

            # if reg_entropy_A > 0:
            #     ent = -(A * (A.clamp_min(1e-8)).log()).sum(dim=-1).mean()
            #     loss = loss - reg_entropy_A * ent  # NOTA il meno: massimizzi entropia

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            total += x_batch.size(0)

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                correct += (pred == y_batch).sum().item()

        return total_loss / max(1, total), correct / max(1, total)

    # TRAINING alternato
    prototipi = torch.zeros((num_classes, dim_space), device=device)

    for it in range(max_iter):
        prototipi = build_prototypes()  # 1) costruisci prototipi con grafo corrente
        metrics_train = evaluate_hdc_graph(train_loader, prototipi, device, flag='train', bundler=graph_step)
        metrics_test = evaluate_hdc_graph(test_loader, prototipi, device, flag='test', bundler=graph_step)
        print(f"[TRAIN/TEST]  Iter: {it+1}/{max_iter},"
              f" Accuracy : {metrics_train['accuracy']:.4f},"
              f" Precision: {metrics_train['precision']:.4f},"
              f" Recall : {metrics_train['recall']:.4f},"
              f" F1-score : {metrics_train['f1_score']:.4f}"
              f" / Accuracy : {metrics_test['accuracy']:.4f},"
              f" Precision: {metrics_test['precision']:.4f},"
              f" Recall : {metrics_test['recall']:.4f},"
              f" F1-score : {metrics_test['f1_score']:.4f}")


        # 2) allena grafo tenendo fissi i prototipi
        for ep in range(graph_epochs):
            loss, acc = train_graph_one_epoch(prototipi)

        # (opzionale) puoi ricostruire i prototipi subito dopo l'update del grafo
        if getattr(config, "rebuild_prototypes_after_graph", False):
            prototipi = build_prototypes()
            evaluate_hdc_graph(train_loader, prototipi, device, flag='train', bundler=graph_step)
            if test_loader is not None:
                evaluate_hdc_graph(test_loader, prototipi, device, flag='test', bundler=graph_step)

    return prototipi, graph_step


def evaluate_hdc_graph(loader, prototipi, device, flag='test', bundler=None):
    """
    Valuta il modello e calcola le metriche.
    (B,C,H)->(B,H)->cosine vs prototipi
    Se bundler != None, assume input (B,C,H) e calcola bundle (B,H).
    Se bundler == None, assume input già (B,H).
    """
    # print(f'\nStart test on {flag} set...')
    correct = 0
    total = 0
    predictions = []
    labels = []
    if not isinstance(prototipi, Tuple):
        prototipi = prototipi.to(device)
    else:
        prototipi, graph_step = prototipi
        bundler = graph_step
        prototipi.to(device)
        graph_step.to(device)

    if bundler is not None:
        bundler.eval()

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # se ho bundler: trasformo (B,C,H)->(B,H)
            if bundler is not None:
                x_batch, _ = bundler(x_batch)
                x_batch = torch.where(x_batch >= 0,
                                     torch.tensor(1.0, device=x_batch.device),
                                     torch.tensor(-1.0, device=x_batch.device))

            for i in range(x_batch.size(0)):
                high_dim_x = x_batch[i]   # (H,)
                y_true = y_batch[i]

                similarities = cosine_similarity(
                    high_dim_x.detach().cpu().numpy().reshape(1, -1),
                    prototipi.detach().cpu().numpy()
                )
                y_pred = int(np.argmax(similarities))

                correct += (y_pred == y_true.item())
                total += 1

                predictions.append(y_pred)
                labels.append(y_true.item())

    accuracy = correct / max(1, total)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)

    # print(f"Accuracy : {accuracy:.4f}, Precision: {precision:.4f}, Recall : {recall:.4f}, F1-score : {f1:.4f}")
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


