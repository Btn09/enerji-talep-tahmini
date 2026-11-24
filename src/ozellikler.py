"""Gün öncesi tahmin için özellikler. Hedef gün t, bilgi t-1 gününün sonuna kadar.

Her müşterinin değerleri kendi eğitim dönemi ortalamasına bölünüyor; böylece küçük bir dükkânla büyük bir fabrika
aynı modelde öğrenilebiliyor (tek, global model).
"""
import holidays
import numpy as np

GECMIS_GUN = 7
OZELLIK_ADLARI = ([f"dun_{k}_gun_once_ayni_saat" for k in range(1, GECMIS_GUN + 1)]
                  + ["dunku_ortalama", "dunku_en_yuksek", "dunku_en_dusuk", "dunku_son_saat",
                     "son7_ayni_saat_ortalama", "gecen_hafta_ayni_gun_ortalama",
                     "saat", "haftanin_gunu", "ay", "log_olcek", "tatil", "dun_tatil"])


def tatil_mi(tarihler):
    """Portekiz resmi tatilleri (veri Portekiz'den). Tatillerde işyerleri kapalı, tüketim belirgin düşüyor."""
    tatiller = holidays.Portugal(years=sorted(set(tarihler.year)))
    return np.array([t.date() in tatiller for t in tarihler], dtype=np.float32)


def olcekler(d, egitim):
    """Müşteri başına eğitim ortalaması (S,)."""
    return d[:, egitim].mean(axis=(1, 2)) + 1e-6


def ozellik_matrisi(dn, olcek, gunler, hedef_gunler):
    """dn: normalize edilmiş (S, G, 24). hedef_gunler: gün indeksleri (t >= 7).
    Dönüş: X (S*T*24, F), y (S*T*24,)"""
    S = dn.shape[0]
    t = np.asarray(hedef_gunler)
    T = len(t)
    saat = np.broadcast_to(np.arange(24), (S, T, 24))
    parcalar = [dn[:, t - k, :] for k in range(1, GECMIS_GUN + 1)]
    dun = dn[:, t - 1, :]
    parcalar += [np.repeat(dun.mean(-1, keepdims=True), 24, -1), np.repeat(dun.max(-1, keepdims=True), 24, -1),
                 np.repeat(dun.min(-1, keepdims=True), 24, -1), np.repeat(dun[..., -1:], 24, -1)]
    son7 = np.stack([dn[:, t - k, :] for k in range(1, 8)]).mean(0)
    gecen_hafta = dn[:, t - 7, :]
    parcalar += [son7, np.repeat(gecen_hafta.mean(-1, keepdims=True), 24, -1)]
    gun_bilgisi = gunler[t]
    parcalar += [saat,
                 np.broadcast_to(gun_bilgisi.dayofweek.values[None, :, None], (S, T, 24)),
                 np.broadcast_to(gun_bilgisi.month.values[None, :, None], (S, T, 24)),
                 np.broadcast_to(np.log(olcek)[:, None, None], (S, T, 24)),
                 np.broadcast_to(tatil_mi(gun_bilgisi)[None, :, None], (S, T, 24)),
                 np.broadcast_to(tatil_mi(gunler[t - 1])[None, :, None], (S, T, 24))]
    X = np.stack([np.asarray(p, np.float32) for p in parcalar], axis=-1).reshape(-1, len(parcalar))
    y = dn[:, t, :].reshape(-1)
    return X, y


def ag_girdisi(dn, olcek, gunler, hedef_gunler):
    """Sinir ağı için: son 7 günün 168 saati + hedef günün takvim bilgisi. Dönüş X (S*T, 180), y (S*T, 24)."""
    S = dn.shape[0]
    t = np.asarray(hedef_gunler)
    gecmis = np.concatenate([dn[:, t - k, :] for k in range(GECMIS_GUN, 0, -1)], axis=-1)  # (S, T, 168)
    g = gunler[t]
    takvim = np.concatenate([np.eye(7)[g.dayofweek.values],
                             np.stack([np.sin(2 * np.pi * g.month.values / 12),
                                       np.cos(2 * np.pi * g.month.values / 12),
                                       tatil_mi(g), tatil_mi(gunler[t - 1])], -1)], -1)
    takvim = np.broadcast_to(takvim[None], (S, len(t), takvim.shape[-1]))
    olcek_ = np.broadcast_to(np.log(olcek)[:, None, None], (S, len(t), 1))
    X = np.concatenate([gecmis, takvim, olcek_], -1).astype(np.float32).reshape(S * len(t), -1)
    return X, dn[:, t, :].reshape(S * len(t), 24)
