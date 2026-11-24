"""Modelleri eğitir ve test döneminde (Ekim-Aralık 2014, 92 gün) karşılaştırır.

Çalıştırma:  python -m src.egit
"""
import json
import time

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import torch

from . import ag
from .ozellikler import GECMIS_GUN, OZELLIK_ADLARI, ag_girdisi, olcekler, ozellik_matrisi
from .veri import KOK, bolum_maskeleri, gunluk_matris

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELLER = KOK / "models"
RAPOR = KOK / "reports"
GRAFIK = RAPOR / "figures"


def wape(y, tahmin):
    """Ağırlıklı mutlak yüzde hata: toplam mutlak hata / toplam tüketim."""
    return float(np.abs(y - tahmin).sum() / np.abs(y).sum())


def lgbm_egit(X_e, y_e, X_d, y_d, amac="l1", alfa=None, agac=2000, ornek=1.0):
    if ornek < 1.0:
        sec = np.random.default_rng(0).random(len(y_e)) < ornek
        X_e, y_e = X_e[sec], y_e[sec]
    ayar = dict(objective=amac, learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                feature_fraction=0.8, bagging_fraction=0.5, bagging_freq=1, verbose=-1, num_threads=8, seed=0)
    if alfa is not None:
        ayar["alpha"] = alfa
    e = lgb.Dataset(X_e, y_e, feature_name=OZELLIK_ADLARI, categorical_feature=["haftanin_gunu"])
    d = lgb.Dataset(X_d, y_d, reference=e)
    return lgb.train(ayar, e, agac, valid_sets=[d], callbacks=[lgb.early_stopping(100, verbose=False)])


def main():
    for k in (MODELLER, GRAFIK):
        k.mkdir(parents=True, exist_ok=True)
    d, gunler, musteriler = gunluk_matris()
    e_m, d_m, t_m = bolum_maskeleri(gunler)
    olcek = olcekler(d, e_m)
    dn = d / olcek[:, None, None]
    S = len(musteriler)
    gun_e = np.flatnonzero(e_m)[GECMIS_GUN:]
    gun_d, gun_t = np.flatnonzero(d_m), np.flatnonzero(t_m)
    print(f"{S} müşteri | eğitim {len(gun_e)} gün | doğrulama {len(gun_d)} | test {len(gun_t)}")

    gercek = d[:, gun_t, :]                                            # (S, T, 24) kWh
    tahminler, sureler = {}, {}

    # Basit yöntemler: dünün aynı saati, geçen haftanın aynı günü
    tahminler["Dünün aynısı"] = d[:, gun_t - 1, :]
    tahminler["Geçen haftanın aynısı"] = d[:, gun_t - 7, :]

    X_e, y_e = ozellik_matrisi(dn, olcek, gunler, gun_e)
    X_d, y_d = ozellik_matrisi(dn, olcek, gunler, gun_d)
    X_t, _ = ozellik_matrisi(dn, olcek, gunler, gun_t)
    print(f"LightGBM eğitim satırı: {len(y_e):,}")
    bas = time.time()
    lgbm = lgbm_egit(X_e, y_e, X_d, y_d)
    sureler["LightGBM"] = time.time() - bas
    print(f"LightGBM: {lgbm.best_iteration} ağaç, {sureler['LightGBM']:.0f} sn")
    p_lgbm = lgbm.predict(X_t, num_iteration=lgbm.best_iteration).reshape(S, len(gun_t), 24)
    tahminler["LightGBM"] = p_lgbm * olcek[:, None, None]

    # Belirsizlik: 10. ve 90. yüzdelik için ayrı modeller (%80 tahmin aralığı)
    aralik = {}
    for ad, alfa in [("p10", 0.1), ("p90", 0.9)]:
        m = lgbm_egit(X_e, y_e, X_d, y_d, amac="quantile", alfa=alfa, agac=600, ornek=0.3)
        aralik[ad] = m.predict(X_t, num_iteration=m.best_iteration).reshape(S, len(gun_t), 24) * olcek[:, None, None]
        m.save_model(str(MODELLER / f"lgbm_{ad}.txt"), num_iteration=m.best_iteration)
    lgbm.save_model(str(MODELLER / "lgbm.txt"), num_iteration=lgbm.best_iteration)
    del X_e, y_e

    A_e, b_e = ag_girdisi(dn, olcek, gunler, gun_e)
    A_d, b_d = ag_girdisi(dn, olcek, gunler, gun_d)
    A_t, _ = ag_girdisi(dn, olcek, gunler, gun_t)
    bas = time.time()
    model, gecmis = ag.egit(A_e, b_e, A_d, b_d)
    sureler["Sinir ağı"] = time.time() - bas
    print(f"Sinir ağı: {len(gecmis)} tur, {sureler['Sinir ağı']:.0f} sn")
    tahminler["Sinir ağı"] = ag.tahmin(model, A_t).reshape(S, len(gun_t), 24) * olcek[:, None, None]
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, MODELLER / "ag.pt")
    tahminler["LightGBM + sinir ağı ortalaması"] = (tahminler["LightGBM"] + tahminler["Sinir ağı"]) / 2

    np.save(MODELLER / "olcekler.npy", olcek)
    (MODELLER / "musteriler.json").write_text(json.dumps(musteriler))

    # Test döneminde hiç tüketimi olmayan (kapanmış) müşteriler medyana katılmıyor
    aktif = [i for i in range(S) if gercek[i].sum() > 0]
    satirlar = []
    for ad, p in tahminler.items():
        satirlar.append({
            "model": ad,
            "musteri_wape": wape(gercek, p),
            "musteri_wape_medyan": float(np.median([wape(gercek[i], p[i]) for i in aktif])),
            "toplam_yuk_mape": float(np.mean(np.abs(gercek.sum(0) - p.sum(0)) / gercek.sum(0))),
            "egitim_sn": round(sureler.get(ad, 0), 1),
        })
        print(f"{ad:32s} müşteri WAPE %{satirlar[-1]['musteri_wape'] * 100:.2f} | medyan "
              f"%{satirlar[-1]['musteri_wape_medyan'] * 100:.2f} | toplam yük MAPE "
              f"%{satirlar[-1]['toplam_yuk_mape'] * 100:.2f}")
    kapsama = float(((gercek >= aralik["p10"]) & (gercek <= aralik["p90"])).mean())
    genislik = float(((aralik["p90"] - aralik["p10"]).sum() / gercek.sum()))
    print(f"%80 aralığın kapsaması: %{kapsama * 100:.1f} | ortalama genişlik tüketimin %{genislik * 100:.1f}'i")

    en_iyi = min(satirlar, key=lambda s: s["musteri_wape"])["model"]
    gunluk = pd.DataFrame({"tarih": gunler[gun_t].strftime("%Y-%m-%d"),
                           **{ad: [wape(gercek[:, i], p[:, i]) for i in range(len(gun_t))]
                              for ad, p in tahminler.items()}})
    gunluk.to_csv(RAPOR / "gunluk_hata.csv", index=False, float_format="%.4f")
    onem = dict(zip(OZELLIK_ADLARI, lgbm.feature_importance("gain").round(0).tolist()))
    sonuc = {"musteri_sayisi": S, "testte_aktif_musteri": len(aktif), "test_donemi": [str(gunler[gun_t][0].date()), str(gunler[gun_t][-1].date())],
             "modeller": satirlar, "en_iyi": en_iyi,
             "aralik": {"kapsama_80": round(kapsama, 4), "ortalama_genislik_orani": round(genislik, 4)},
             "en_kotu_gunler": gunluk.nlargest(5, en_iyi)[["tarih", en_iyi]].values.tolist(),
             "lgbm_ozellik_onemi": onem}
    with open(RAPOR / "sonuclar.json", "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    pd.DataFrame(satirlar).to_csv(RAPOR / "model_karsilastirma.csv", index=False, float_format="%.4f")

    grafikler(gercek, tahminler, aralik, gunler[gun_t], musteriler, gunluk, en_iyi)


def grafikler(gercek, tahminler, aralik, tarihler, musteriler, gunluk, en_iyi):
    saatler = pd.date_range(tarihler[0], periods=len(tarihler) * 24, freq="h")
    hafta = slice(14 * 24, 21 * 24)

    fig, eksen = plt.subplots(figsize=(11, 4))
    eksen.plot(saatler[hafta], gercek.sum(0).ravel()[hafta] / 1000, color="black", lw=1.5, label="Gerçek")
    for ad in ["Geçen haftanın aynısı", "LightGBM", "Sinir ağı"]:
        eksen.plot(saatler[hafta], tahminler[ad].sum(0).ravel()[hafta] / 1000, lw=1, label=ad)
    eksen.set_ylabel("Toplam yük (MWh)")
    eksen.set_title("315 müşterinin toplam saatlik tüketimi: bir test haftası")
    eksen.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(GRAFIK / "toplam_yuk.png", dpi=140)
    plt.close(fig)

    i = int(np.argsort(gercek.mean((1, 2)))[len(musteriler) // 2])  # ortanca büyüklükte bir müşteri
    fig, eksen = plt.subplots(figsize=(11, 4))
    eksen.fill_between(saatler[hafta], aralik["p10"][i].ravel()[hafta], aralik["p90"][i].ravel()[hafta],
                       alpha=0.3, label="%80 tahmin aralığı")
    eksen.plot(saatler[hafta], gercek[i].ravel()[hafta], color="black", lw=1.2, label="Gerçek")
    eksen.plot(saatler[hafta], tahminler["LightGBM"][i].ravel()[hafta], lw=1, label="LightGBM")
    eksen.set_ylabel("kWh")
    eksen.set_title(f"Tek müşteri ({musteriler[i]}): tahmin ve belirsizlik")
    eksen.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(GRAFIK / "tek_musteri.png", dpi=140)
    plt.close(fig)

    fig, eksen = plt.subplots(figsize=(11, 3.8))
    for ad in ["Geçen haftanın aynısı", en_iyi]:
        eksen.plot(pd.to_datetime(gunluk["tarih"]), gunluk[ad] * 100, label=ad)
    eksen.set_ylabel("Günlük WAPE (%)")
    eksen.set_title("Test dönemi boyunca günlük hata")
    eksen.legend(fontsize=8)
    eksen.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAFIK / "gunluk_hata.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
