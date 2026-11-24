"""UCI ElectricityLoadDiagrams20112014: Portekiz'de 370 müşterinin 15 dakikalık elektrik tüketimi (kW).

Saatlik kWh'e çevirip 2012-2014 arasını kullanıyorum. 2012 başında henüz bağlanmamış müşteriler çıkarılıyor.
Görev: her gün sonunda, ertesi günün 24 saatini tahmin etmek (gün öncesi piyasası gibi).

Çalıştırma:  python -m src.veri   (data/indirme/LD2011_2014.txt'den data/saatlik.csv.gz üretir)
"""
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parent.parent
HAM = KOK / "data" / "indirme" / "LD2011_2014.txt"
SAATLIK = KOK / "data" / "saatlik.csv.gz"

BASLANGIC, BITIS = "2012-01-01", "2014-12-31 23:00"
DOGRULAMA_BASI, TEST_BASI = "2014-07-01", "2014-10-01"


def saatlik_hazirla():
    df = pd.read_csv(HAM, sep=";", decimal=",", index_col=0, parse_dates=True)
    # Değerler 15 dakikalık ortalama kW; 4'e bölünce o 15 dakikadaki kWh. Zaman damgası aralığın sonu.
    df = (df / 4).resample("h", closed="right", label="left").sum()
    df = df.loc[BASLANGIC:BITIS]
    ilk_hafta = df.loc["2012-01-01":"2012-01-07"]
    df = df.loc[:, (ilk_hafta > 0).mean() > 0.9]
    df.to_csv(SAATLIK, float_format="%.3f")
    print(f"{df.shape[1]} müşteri, {len(df)} saat -> {SAATLIK}")


def gunluk_matris(df=None):
    """(müşteri, gün, 24) dizisi, gün tarihleri ve müşteri adları."""
    if df is None:
        df = pd.read_csv(SAATLIK, index_col=0, parse_dates=True)
    gunler = pd.DatetimeIndex(sorted(set(df.index.normalize())))
    deger = df.values.T.reshape(df.shape[1], len(gunler), 24)
    return deger.astype(np.float32), gunler, list(df.columns)


def bolum_maskeleri(gunler):
    egitim = gunler < DOGRULAMA_BASI
    dogrulama = (gunler >= DOGRULAMA_BASI) & (gunler < TEST_BASI)
    test = gunler >= TEST_BASI
    return egitim, dogrulama, test


if __name__ == "__main__":
    saatlik_hazirla()
