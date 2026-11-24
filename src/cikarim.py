"""Kayıtlı modellerle tek müşteri için ertesi gün tahmini. API bunu kullanıyor."""
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from .ag import GunOncesiAg
from .ozellikler import GECMIS_GUN, ag_girdisi, ozellik_matrisi
from .veri import KOK

MODEL_KLASORU = Path(os.environ.get("MODEL_KLASORU", KOK / "models"))


class Tahminci:
    def __init__(self):
        self.lgbm = lgb.Booster(model_file=str(MODEL_KLASORU / "lgbm.txt"))
        self.p10 = lgb.Booster(model_file=str(MODEL_KLASORU / "lgbm_p10.txt"))
        self.p90 = lgb.Booster(model_file=str(MODEL_KLASORU / "lgbm_p90.txt"))
        self.ag = GunOncesiAg()
        self.ag.load_state_dict(torch.load(MODEL_KLASORU / "ag.pt", map_location="cpu"))
        self.ag.eval()
        self.olcekler = np.load(MODEL_KLASORU / "olcekler.npy")
        self.musteriler = json.loads((MODEL_KLASORU / "musteriler.json").read_text())
        self.sira = {m: i for i, m in enumerate(self.musteriler)}

    def tahmin_et(self, musteri, hedef_tarih, son_7_gun):
        """son_7_gun: hedef günden önceki 7 günün saatlik tüketimi (168 değer, kWh)."""
        olcek = self.olcekler[self.sira[musteri]:self.sira[musteri] + 1]
        gecmis = np.asarray(son_7_gun, np.float32).reshape(1, GECMIS_GUN, 24) / olcek[:, None, None]
        dn = np.concatenate([gecmis, np.zeros((1, 1, 24), np.float32)], axis=1)  # son gün: hedef (bilinmiyor)
        gunler = pd.date_range(end=pd.Timestamp(hedef_tarih), periods=GECMIS_GUN + 1, freq="D")

        X, _ = ozellik_matrisi(dn, olcek, gunler, [GECMIS_GUN])
        A, _ = ag_girdisi(dn, olcek, gunler, [GECMIS_GUN])
        with torch.no_grad():
            p_ag = self.ag(torch.from_numpy(A)).numpy().ravel()
        orta = (self.lgbm.predict(X) + p_ag) / 2 * olcek[0]
        alt = self.p10.predict(X) * olcek[0]
        ust = self.p90.predict(X) * olcek[0]
        # Ayrı modellerden geldikleri için aralık tahmini kapsamayabilir; sıralamayı koru
        orta = np.clip(orta, 0, None)
        alt, ust = np.clip(np.minimum(alt, orta), 0, None), np.maximum(ust, orta)
        return orta, alt, ust
