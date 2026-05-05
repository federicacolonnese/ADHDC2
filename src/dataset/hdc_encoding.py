import torch
import torchhd
from typing import List


class ContinuousItemMemory:
    """
    Classe per la codifica di serie temporali utilizzando Continuous Item Memory (CiM)
    """
    def __init__(self,
                 num_levels: int = 10,
                 dim_hv: int = 10):
        self.num_levels = num_levels  # numero di livelli di quantizzazione
        self.dim_hv = dim_hv
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

        # Inizializzazione CiM progressiva

        # Inizializzo HV0 per il livello più basso della cim
        H0 = torchhd.random(1, self.dim_hv, vsa="MAP").squeeze(0).float()

        # Inizializzo HV_last per il livello più alto, flippando la metà dei bit di H0
        half = self.dim_hv // 2
        perm = torch.randperm(self.dim_hv, device=sample_values.device)  # cambio casuale degli indici
        flip_idx = perm[:half]  # prendo i primi D/2 indici da flippare
        H_last = H0.clone()
        H_last[flip_idx] *= -1  # flipping dei bit in modo casuale

        # Creazione progressiva dei livelli flippando sempre meno bit ad ogni step
        self.level_hv = torch.zeros((self.num_levels, self.dim_hv), device=sample_values.device)
        self.level_hv[0] = H0
        self.level_hv[self.num_levels-1] = H_last

        # Numero di bit da flippare a ogni step
        delta = half // (self.num_levels - 1)

        flip_blocks = flip_idx.split(delta)
        H_prev = H0.clone()

        for j in range(1, self.num_levels - 1):
            H_j = H_prev.clone()
            idx_to_flip = flip_blocks[j - 1]
            H_j[idx_to_flip] *= -1
            self.level_hv[j] = H_j
            H_prev = H_j


        # # Prova senza progressione HV (to delete or modify)
        # self.level_hv = torchhd.random(self.num_levels, self.dim_hv, vsa="MAP")
    
    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """
        Codifica una serie di valori continui in HV utilizzando la CiM
        values: Tensor di forma (T,) con valori continui
        Ritorna: Tensor di forma (T, dim_hv) con HV codificati
        """
        T = values.shape[0]
        # print(f"CiM: Encoding {T} values into HVs...")
        dim_hv = self.dim_hv
        hv_encoded = torch.zeros((T, dim_hv), device=values.device) #inizializzo tensor vuoto per gli HV codificati

        # Mappa i valori continui ai livelli della CiM
        for t in range(T):
            val = values[t].item() # prendo il valore scalare

            # Normalizza il valore nell'intervallo [0, 1] RENDERLO PARAMETRICO QUANDO CALCOLO NUM LEVELS
            normalized = (val - self.low) / (self.high - self.low)

            # Mappa al livello corrispondente
            level_idx = int(normalized * (self.num_levels - 1))
            level_idx = max(0, min(self.num_levels - 1, level_idx))  # clamp

            hv_encoded[t] = self.level_hv[level_idx] #assegna l'HV corrispondente al livello
        return hv_encoded 


class HD_data_encoding:

    """
    Encoder iperdimensionale per nodi (canali) secondo la formalizzazione:

    - FORMALIZZAZIONE - creazione CIM e IM
    - FORMALIZZAZIONE - creazione HV FINALI

    Ogni nodo = 1 canale = 1 sequenza temporale (+ opzionale matrice di feature).
    """

    def __init__(self,
                 num_timesteps: int,
                 num_levels: int = 10,
                 dim_hv: int = 100,
                 vsa: str = "MAP",
                 channel_bundling: bool = False,):

        self.num_timesteps = num_timesteps
        self.dim_hv = dim_hv
        self.vsa = vsa
        self.num_levels = num_levels
        self.channel_bundling = channel_bundling
        self.cim_per_channel = list()
            
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
                      list_cim_per_channel: List[ContinuousItemMemory]) -> torch.Tensor:
        """
        Codifica l'intero dataset di canali (nodi).
        Ogni canale ha la propria CiM (per dati eterogenei).
        - data: shape (N, T)
        
        """
        C = sample.shape[0]
        # print(f"Encoding dataset with {C} channels...")

        sample_hv = torch.zeros((C, self.dim_hv), device=sample.device)

        for i in range(C):
            channel_values = sample[i]  # (T,)

            # 1) CREA UNA NUOVA CiM per questo canale
            cim = list_cim_per_channel[i]

            # 3) encoding del canale usando la sua CiM
            hv_channel = self.encode_channel(cim,
                                             channel_values)
            sample_hv[i] = hv_channel

        return sample_hv

    def encode_dataset(self,
                       dataset: torch.Tensor,
                       channel_bundling: bool) -> torch.Tensor:
        """
        Codifica l'intero dataset. Ogni canale ha la propria CiM (per dati eterogenei).
        - dataset: shape (N, C, T)
        """
        N, C, T = dataset.shape
        # print(f"Encoding dataset with shape {N, C, T} ...")

        # Initialize and create a CIM for channel
        for i in range(C):
            curr_cim = ContinuousItemMemory(num_levels=self.num_levels,
                                            dim_hv=self.dim_hv)  # create cim
            channel_values = dataset[:, i, :]
            curr_cim.fit(channel_values)  # train cim

            self.cim_per_channel.append(curr_cim)

        # Encode data in HV
        hv_dataset = torch.zeros((N, C, self.dim_hv),
                                 device=dataset.device)
        for sample in range(N):
            curr_sample = dataset[sample, :, :]
            hv_dataset[sample] = self.encode_sample(curr_sample,
                                                    self.cim_per_channel)

        # Channels bundling
        if channel_bundling:
            hv_dataset_no_channel = torch.zeros((N, self.dim_hv), device=dataset.device)
            for sample in range(N):
                curr_sample = hv_dataset[sample, :, :]

                # Bundling: operazione di somma binaria (XOR)
                hv_dataset_no_channel[sample] = torch.where(torchhd.multiset(curr_sample) >= 0, torch.tensor(1.0),
                                                            torch.tensor(-1.0))
            return hv_dataset_no_channel
        else:
            return hv_dataset


