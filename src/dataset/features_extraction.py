from typing import List

import numpy as np
import torch
import torchhd

from src.config import Config


class TimeSeriesFeatureExtractor:
    def __init__(self, config, normalize=False):
        """
        Estrae feature multi-risoluzione da una serie temporale.

        Per ogni n in n_values:
        - divide l'asse temporale in n segmenti contigui (con punti di confine condivisi)
        - calcola il seno dell'angolo di crescita di ciascun segmento

        normalize:
        - se True: z-score prima delle pendenze (utile per confrontabilità tra serie)
        """
        self.config = config
        n_values = config.n_values_feat
        self.n_values = tuple(int(n) for n in n_values)
        if len(self.n_values) == 0:
            raise ValueError("n_values non può essere vuoto.")
        if any(n < 1 for n in self.n_values):
            raise ValueError("Tutti i valori in n_values devono essere >= 1.")
        self.normalize = bool(normalize)

    def _validate_and_prepare(self, series) -> np.ndarray:
        x = np.asarray(series, dtype=float).reshape(-1)
        if x.size == 0:
            raise ValueError("La serie è vuota.")
        if not np.isfinite(x).all():
            raise ValueError("La serie contiene NaN/inf: pulisci i dati prima.")
        if self.normalize:
            m = x.mean()
            s = x.std()
            if s > 0:
                x = (x - m) / s
            else:
                x = x * 0.0
        return x

    def _boundaries(self, T: int, n: int) -> np.ndarray:
        """
        Calcola n+1 indici di confine (breakpoints) tra 0 e T-1 inclusi,
        distribuendo gli intervalli (T-1) in modo più equo possibile tra n segmenti.

        Esempio: T=8 (intervalli=7), n=3 -> intervalli: [3,2,2]
        boundaries: [0,3,5,7]
        segmenti: [0..3], [3..5], [5..7]
        """
        if T <= 0:
            raise ValueError("T deve essere > 0.")
        if n < 1:
            raise ValueError("n deve essere >= 1.")

        intervals = max(T - 1, 0)
        base = intervals // n
        rem = intervals % n

        b = [0]
        for j in range(n):
            step = base + (1 if j < rem else 0)
            b.append(b[-1] + step)

        # per costruzione l'ultimo deve essere T-1
        b[-1] = T - 1
        return np.asarray(b, dtype=int)

    def compute_segment_slopes(self, series, n: int) -> np.ndarray:
        """
        Ritorna le pendenze dei n segmenti contigui:
        slope_j = (y[end] - y[start]) / (end - start)

        Se end == start (segmento degenerato), slope = 0.
        """
        x = self._validate_and_prepare(series)
        T = len(x)
        b = self._boundaries(T, n)

        slopes = np.zeros(n, dtype=float)
        for j in range(n):
            s = int(b[j])
            e = int(b[j + 1])
            dx = e - s
            if dx <= 0:
                slopes[j] = 0.0
            else:
                slopes[j] = float((x[e] - x[s]) / dx)
        return slopes

    def compute_sine_growth_angles(self, series, n: int) -> np.ndarray:
        """
        sin(angle) con angle = arctan(slope) => sin(angle) = slope / sqrt(1 + slope^2)
        """
        slopes = self.compute_segment_slopes(series, n)
        return slopes / np.sqrt(1.0 + slopes * slopes)

    def extract_features_n_sins(self, series):
        """
        Ritorna:
        - raw: shape (T,)
        - features: shape (sum(n_values),) (concatenazione dei seni degli angoli)
        - list_of_pos_classes: [n1, n2, ...]
        """
        raw = np.asarray(series, dtype=float).reshape(-1)

        all_feats = []
        list_of_pos_classes = []
        for n in self.n_values:
            f = self.compute_sine_growth_angles(raw, n)
            all_feats.append(f)
            list_of_pos_classes.append(len(f))  # = n

        features = np.concatenate(all_feats, axis=0) if all_feats else np.array([], dtype=float)
        return raw, features, list_of_pos_classes

    def build_piecewise_linear(self, series, n: int, continuous: bool = True, use_prepared: bool = False):
        """
        Costruisce una spezzata y_hat lunga T con slope costante in ciascun segmento,
        usando segmenti contigui con punti di confine condivisi.

        continuous=True:
        - la spezzata è continua (il segmento j parte dal valore di fine del segmento j-1)

        use_prepared=True:
        - usa la serie "preparata" (_validate_and_prepare), quindi se normalize=True
          la visualizzazione è nello stesso spazio delle feature.
        """
        x_plot = self._validate_and_prepare(series) if use_prepared else np.asarray(series, dtype=float).reshape(-1)
        T = len(x_plot)
        b = self._boundaries(T, n)
        slopes = self.compute_segment_slopes(series if not use_prepared else x_plot, n)

        y_hat = np.empty(T, dtype=float)
        y_hat[:] = np.nan

        # inizializza primo confine
        y_hat[b[0]] = x_plot[b[0]]

        for j in range(n):
            s = int(b[j])
            e = int(b[j + 1])
            slope = float(slopes[j])

            # assicura ancoraggio al punto di start
            if j == 0:
                y_hat[s] = x_plot[s]
            else:
                if continuous:
                    # dovrebbe già essere stato scritto come end del segmento precedente
                    if not np.isfinite(y_hat[s]):
                        y_hat[s] = y_hat[int(b[j] - 0)] if s > 0 and np.isfinite(y_hat[s - 1]) else x_plot[s]
                else:
                    y_hat[s] = x_plot[s]

            if e == s:
                continue  # segmento degenerato: solo un punto

            # per evitare overwrite del confine condiviso, dal secondo segmento in poi parto da s+1
            t0 = s if j == 0 else s + 1
            anchor = float(y_hat[s])

            for t in range(t0, e + 1):
                y_hat[t] = anchor + slope * (t - s)

        # fallback se resta qualche NaN
        nan_mask = ~np.isfinite(y_hat)
        if nan_mask.any():
            y_hat[nan_mask] = x_plot[nan_mask]

        return y_hat, b

    def plot_piecewise(
        self,
        series,
        n_values=None,
        continuous: bool = True,
        degrees: bool = True,
        subplots: bool = True,
        figsize=(11, 3),
        show_boundaries: bool = True,
        show: bool = True,
        use_prepared: bool = False,
    ):
        """
        Visualizza la serie originale e la sua approssimazione spezzata per ciascun n.

        - subplots=True: più righe (una per n), quindi "plot orizzontali a diversa scala"
        - show_boundaries=True: disegna linee verticali sui confini dei segmenti
        """
        import matplotlib.pyplot as plt

        x_plot = self._validate_and_prepare(series) if use_prepared else np.asarray(series, dtype=float).reshape(-1)
        T = len(x_plot)
        if T == 0:
            raise ValueError("La serie è vuota.")

        n_values = tuple(self.n_values if n_values is None else (int(v) for v in n_values))

        def _angles_text(slopes):
            ang = np.arctan(slopes)
            return np.degrees(ang) if degrees else ang

        figs = []

        if subplots:
            fig, axes = plt.subplots(len(n_values), 1, sharex=True, figsize=(figsize[0], figsize[1] * len(n_values)))
            if len(n_values) == 1:
                axes = [axes]

            for ax, n in zip(axes, n_values):
                y_hat, b = self.build_piecewise_linear(series, n, continuous=continuous, use_prepared=use_prepared)
                slopes = self.compute_segment_slopes(series if not use_prepared else x_plot, n)
                angles = _angles_text(slopes)

                ax.plot(x_plot, label="serie")
                ax.plot(y_hat, label=f"spezzata (n={n})")

                if show_boundaries:
                    for k in b[1:-1]:
                        ax.axvline(int(k), linestyle="--", linewidth=1)

                ax.set_title(f"n={n} | angoli ({'deg' if degrees else 'rad'}): {np.round(angles, 3)}")
                ax.legend()

            figs.append(fig)
            if show:
                plt.show()
            return figs

        # figure separate
        for n in n_values:
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()

            y_hat, b = self.build_piecewise_linear(series, n, continuous=continuous, use_prepared=use_prepared)
            slopes = self.compute_segment_slopes(series if not use_prepared else x_plot, n)
            angles = _angles_text(slopes)

            ax.plot(x_plot, label="serie")
            ax.plot(y_hat, label=f"spezzata (n={n})")

            if show_boundaries:
                for k in b[1:-1]:
                    ax.axvline(int(k), linestyle="--", linewidth=1)

            ax.set_title(f"n={n} | angoli ({'deg' if degrees else 'rad'}): {np.round(angles, 3)}")
            ax.legend()

            figs.append(fig)
            if show:
                plt.show()

        return figs

    def compute_ema(self, series):
        ema = [series[0]]
        for val in series[1:]:
            ema.append(self.config.alpha * val + (1 - self.config.alpha) * ema[-1])

        return np.array(ema)

    def compute_fft_topk(self, series):
        fft_vals = np.fft.fft(series)
        mag = np.abs(fft_vals)[:len(fft_vals) // 2]  # metà spettro
        topk_idx = mag.argsort()[-self.config.k_fft:][::-1]
        topk_vals = mag[topk_idx]

        return topk_vals  # vettore di dimensione k_fft

    def extract_features_ema_fft(self, series):
        """
        Estrae:
        - valore grezzo (serie)
        - feature EMA
        - feature FFT replicate
        Ritorna:
        - raw: shape (T,) valori grezzi
        - features: shape (T, s) con le sole feature derivate
        """
        T = len(series)
        ema = self.compute_ema(series)
        fft_vals = self.compute_fft_topk(series)
        features = np.concatenate((ema, fft_vals))
        list_of_pos_classes = [len(ema), len(fft_vals)]
        return series, features, list_of_pos_classes

#
#
#
#
# class TimeSeriesFeatureExtractor:
#     def __init__(self, alpha=0.3, k_fft=2):
#         """
#         Estrae feature da una serie temporale:
#         - EMA con fattore di smoothing alpha
#         - FFT top-k componenti
#         """
#
#         print("TIME SERIES FEATURE EXTRACTION ENABLED")
#
#
#     def compute_ema(self, series):
#         ema = [series[0]]
#         for val in series[1:]:
#             ema.append(self.alpha * val + (1 - self.alpha) * ema[-1])
#
#         return np.array(ema)
#
#     def compute_fft_topk(self, series):
#         fft_vals = np.fft.fft(series)
#         mag = np.abs(fft_vals)[:len(fft_vals) // 2]  # metà spettro
#         topk_idx = mag.argsort()[-self.k_fft:][::-1]
#         topk_vals = mag[topk_idx]
#
#         return topk_vals  # vettore di dimensione k_fft
#
#     def extract_features(self, series):
#         """
#         Estrae:
#         - valore grezzo (serie)
#         - feature EMA
#         - feature FFT replicate
#         Ritorna:
#         - raw: shape (T,) valori grezzi
#         - features: shape (T, s) con le sole feature derivate
#         """
#         T = len(series)
#         ema = self.compute_ema(series)
#         fft_vals = self.compute_fft_topk(series)
#         #print("EMA SIZE:", ema.shape, type(ema))
#         #print("FFT VALS SIZE:", fft_vals.shape, type(fft_vals))
#         features = np.concatenate((ema, fft_vals))
#         #print("FEATURES SIZE:", features.shape, type(features))
#         list_of_pos_classes = [len(ema), len(fft_vals)]
#         #print(f"Extracted features shape: {features.shape}, list_of_pos_classes: {list_of_pos_classes}")
#         return series, features, list_of_pos_classes
#
#         #fft_rep = np.tile(fft_vals, (T, 1)) quest tecnicamente non ci serve perchè non vogliamo più avere i valori ripetuti
#         #features = np.column_stack([ema, fft_vals])  # solo feature derivate (senza valore grezzo)
#
#         # ema_size = np.expand_dims(ema, axis=-1).shape[-1]
#         # fft_size = fft_vals.shape[-1]
#         # list_of_pos_classes = [ema_size, fft_size]
#         # print(f"Extracted features shape: {features.shape}, list_of_pos_classes: {list_of_pos_classes}")
#         # return series, features, list_of_pos_classes


class FeaturesContinuousItemMemory:
    """
    Classe per la codifica di serie temporali utilizzando Continuous Item Memory (CiM)
    """

    def __init__(self,
                 num_levels: int = 10,
                 dim_hv: int = 10,
                 dim_f: int = 1):
        self.num_levels = num_levels  # numero di livelli di quantizzazione
        self.dim_hv = dim_hv
        self.dim_f = dim_f
        self.low = None
        self.high = None

    def fit(self,
            sample_values: torch.Tensor):

        # Calcolo range
        self.low = float(sample_values.flatten().min())
        self.high = float(sample_values.flatten().max())

        # se high e low sono uguali
        if self.high == self.low:
            self.high = self.low + 1e-6

        # Prova senza progressione HV (to delete or modify)
        self.level_hv = torchhd.random(self.num_levels, self.dim_hv, vsa="MAP")

    def encode(self, values: torch.Tensor) -> torch.Tensor:

        """
        Codifica una serie di valori continui in HV utilizzando la CiM
        values: Tensor di forma (T,) con valori continui
        Ritorna: Tensor di forma (T, dim_hv) con HV codificati
        """

        T = values.shape[0]
        # print(f"CiM: Encoding {T} values into HVs...")
        dim_hv = self.dim_hv
        hv_encoded = torch.zeros((T, dim_hv), device=values.device)  # inizializzo tensor vuoto per gli HV codificati

        # Mappa i valori continui ai livelli della CiM
        for t in range(T):
            val = values[t].item()  # prendo il valore scalare

            # Normalizza il valore nell'intervallo [0, 1] RENDERLO PARAMETRICO QUANDO CALCOLO NUM LEVELS
            normalized = (val - self.low) / (self.high - self.low)

            # Mappa al livello corrispondente
            level_idx = int(normalized * (self.num_levels - 1))
            level_idx = max(0, min(self.num_levels - 1, level_idx))  # clamp

            hv_encoded[t] = self.level_hv[level_idx]  # assegna l'HV corrispondente al livello
        return hv_encoded


class FeaturesHDDataEncoding:
    """
    Encoder iperdimensionale per nodi (canali) secondo la formalizzazione:

    - FORMALIZZAZIONE - creazione CIM e IM
    - FORMALIZZAZIONE - creazione HV FINALI

    Ogni nodo = 1 canale = 1 sequenza temporale (+ opzionale matrice di feature).
    """

    def __init__(self,
                 list_of_pos_classes,
                 num_levels: int = 10,
                 dim_hv: int = 100,
                 vsa: str = "MAP",
                 channel_bundling: bool = False, ):

        self.list_of_pos_classes = list_of_pos_classes
        self.dim_hv = dim_hv
        self.vsa = vsa
        self.num_levels = num_levels
        self.channel_bundling = channel_bundling
        self.cim_per_channel_per_feat = list(list())
        
        # Crea key vectors per ogni tipo di feature (EMA, FFT, ecc.)
        self.feature_type_keys = torchhd.random(len(list_of_pos_classes), dim_hv, vsa=vsa)
        print(f"Created {len(list_of_pos_classes)} feature type keys")

    def encode_channel(self,
                       cim,
                       channel_values: torch.Tensor) -> torch.Tensor:
        # Channel_values shape: (T,)
        T = channel_values.shape[0]
        # Codifica i valori del canale utilizzando la CiM
        hv_encoded_raw = cim.encode(channel_values)  # shape: (T, D)
        # print(f"Encoding channel of length {T}...")
        # Faccio permutazione
        permuted = []
        for t in range(T):
            # torchhd.permute accetta un singolo HV e lo ruota di 'shifts'
            hv_t_perm = torchhd.permute(hv_encoded_raw[t], shifts=t)
            permuted.append(hv_t_perm)

        permuted = torch.stack(permuted, dim=0)  # (T, D)

        # Bundling degli HV permutati per ottenere l'HV finale del canale
        hv_channel = torchhd.multiset(permuted)  # shape: (D,)
        hv_channel = torch.where(hv_channel >= 0, torch.tensor(1.0), torch.tensor(-1.0))
        return hv_channel

    def encode_sample(self,
                      sample: torch.Tensor,
                      list_cim_per_channel: List[FeaturesContinuousItemMemory]) -> torch.Tensor:
        """
        Codifica l'intero dataset di canali (nodi).
        Ogni canale ha la propria CiM per ogni tipo di feature.
        - sample: shape (C, F) dove F = sum(list_of_pos_classes)
        """
        C = sample.shape[0]
        num_feat_types = len(self.list_of_pos_classes)
        
        sample_hv = torch.zeros((C, self.dim_hv), device=sample.device)
        
        for i in range(C):
            channel_values = sample[i]  # (F,) 
            
            # Lista di HV per ogni tipo di feature
            hv_per_feat_type = []
            cumulate_feat_dim = 0
            
            # Itera su ogni tipo di feature (EMA, FFT, ecc.)
            for feat_idx, dim_f in enumerate(self.list_of_pos_classes):
                # Calcola l'indice della CIM corretta per questo canale e tipo di feature
                cim_idx = i * num_feat_types + feat_idx
                cim = list_cim_per_channel[cim_idx]
                
                # Estrai il blocco di features corrispondente
                feat_block = channel_values[cumulate_feat_dim:cumulate_feat_dim + dim_f]
                
                # Codifica ogni valore del blocco con la sua CIM (SENZA permutazione temporale)
                # cim.encode ritorna (dim_f, dim_hv)
                hv_feat = cim.encode(feat_block)  # shape: (dim_f, dim_hv)
                
                # Bundle dei valori di questo tipo di feature
                hv_feat_bundled = torchhd.multiset(hv_feat)  # (dim_f, dim_hv) → (dim_hv,)
                hv_feat_bundled = torch.where(hv_feat_bundled >= 0, torch.tensor(1.0), torch.tensor(-1.0))
                
                # Bind con la key del tipo di feature
                feature_key = self.feature_type_keys[feat_idx]
                hv_feat_bound = torchhd.bind(hv_feat_bundled, feature_key)  # (dim_hv,)
                
                hv_per_feat_type.append(hv_feat_bound)
                cumulate_feat_dim += dim_f
            
            # Bundle finale di tutti i tipi di feature
            hv_per_feat_type = torch.stack(hv_per_feat_type, dim=0)  # (num_feat_types, dim_hv)
            hv_channel = torchhd.multiset(hv_per_feat_type)  # (num_feat_types, dim_hv) → (dim_hv,)
            hv_channel = torch.where(hv_channel >= 0, torch.tensor(1.0), torch.tensor(-1.0))
            sample_hv[i] = hv_channel

        return sample_hv

    def encode_dataset(self,
                       dataset: torch.Tensor,
                       channel_bundling: bool) -> torch.Tensor:
        """
        Codifica l'intero dataset. Ogni canale ha la propria CiM (per dati eterogenei).
        - dataset: shape (N, C, T)
        """
        #N, C, T, F = dataset.shape adesso non abbiamo più questo shape perchè non ho per ogni timestep tutte le feature (?) 
        #CHECK se questo è corretto 
        N,C,F = dataset.shape

        # Features = [F_0, F_1, F_2, ..., F_m], with
        # feat_class = torch.tensor([0,1,1,2,3,4,4,4,...,c])
        # classes = torch.unique(feat_class)
        #

        # Initialize and create a CIM for channel
        for i in range(C):
            cumulate_feat_dim=0
            for dim_f in self.list_of_pos_classes: 
                #calcolo la cim per ogni tipo di features 
                curr_cim = FeaturesContinuousItemMemory(num_levels=self.num_levels,
                                                        dim_hv=self.dim_hv,
                                                        dim_f=dim_f)  # create cim
                channel_values = dataset[:, i, cumulate_feat_dim:cumulate_feat_dim+dim_f]
                curr_cim.fit(channel_values)  # train cim

                self.cim_per_channel_per_feat.append(curr_cim)  # [cim_C0F0, cim_C0F1, cimC1F0, cim_C1F1, ..., cim_CCFF]
                cumulate_feat_dim += dim_f
        
        # Debug: ispeziona le CIM create
        print(f"\n=== DEBUG CIM PER CHANNEL PER FEAT ===")
        print(f"Numero totale di CIM create: {len(self.cim_per_channel_per_feat)}")
        for idx, cim in enumerate(self.cim_per_channel_per_feat):
            print(f"CIM {idx}: dim_f={cim.dim_f}, num_levels={cim.num_levels}, low={cim.low:.4f}, high={cim.high:.4f}")
        print(f"=====================================\n")
        
        # Encode data in HV
        #inizializzo dataset HV finale per features 
        hv_dataset = torch.zeros((N, C, self.dim_hv),
                                 device=dataset.device)
        print("shape HV dataset", hv_dataset.shape)
        for sample in range(N):
            curr_sample = dataset[sample, :, :] #CANALI, FEAT
            hv_dataset[sample] = self.encode_sample(curr_sample,
                                                    self.cim_per_channel_per_feat)

        # Channels bundling
        if channel_bundling:
            hv_dataset_no_channel = torch.zeros((N, self.dim_hv),
                                                device=dataset.device)
            for sample in range(N):
                curr_sample = hv_dataset[sample, :, :]

                # Bundling: operazione di somma binaria (XOR)
                hv_dataset_no_channel[sample] = torch.where(torchhd.multiset(curr_sample) >= 0, torch.tensor(1.0),
                                                            torch.tensor(-1.0))
            return hv_dataset_no_channel
        else:
            return hv_dataset


if __name__ == "__main__":
    series = [1, 2, 1.5, 3, 2.8, 4, 3.5, 5, 4, 3.7,1.2, 1.1,1.2,1.3,0.7,0.4,0.3,4.5,6.7,4,4,3,5,7,0.2,0.3,0.9,3.4,5,6,7,8,8,8,8,8]
    config = Config()
    config.n_values_feat = (2, 3, 4, 5, 6, 7, 8)
    ext = TimeSeriesFeatureExtractor(config=config, normalize=False)

    raw, features, pos = ext.extract_features_n_sins(series)

    # Plot separati (uno per n)
    ext.plot_piecewise(series, subplots=False, continuous=True, degrees=True)

    # Oppure: tutti nello stesso “stack” (più righe)
    ext.plot_piecewise(series, subplots=True, continuous=True, degrees=True)