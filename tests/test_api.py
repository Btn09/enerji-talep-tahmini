import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import app

ORNEK = json.loads((Path(__file__).parent / "ornek.json").read_text())


def istek(**degisiklik):
    return {k: v for k, v in {**ORNEK, **degisiklik}.items() if k != "gercek"}


@pytest.fixture(scope="module")
def istemci():
    with TestClient(app) as c:
        yield c


def test_musteriler(istemci):
    m = istemci.get("/musteriler").json()
    assert len(m) == 315 and ORNEK["musteri"] in m


def test_tahmin_gercege_yakin(istemci):
    c = istemci.post("/tahmin", json=istek()).json()
    tahmin = np.array([s["tahmin"] for s in c["saatler"]])
    gercek = np.array(ORNEK["gercek"])
    assert len(tahmin) == 24
    assert np.abs(tahmin - gercek).sum() / gercek.sum() < 0.15
    assert abs(c["gunluk_toplam"] - tahmin.sum()) < 0.1


def test_aralik_sirali(istemci):
    for s in istemci.post("/tahmin", json=istek()).json()["saatler"]:
        assert 0 <= s["alt"] <= s["tahmin"] <= s["ust"]


def test_tuketim_iki_katina_cikarsa_tahmin_de_artar(istemci):
    az = istemci.post("/tahmin", json=istek()).json()["gunluk_toplam"]
    cok = istemci.post("/tahmin", json=istek(son_7_gun=[2 * x for x in ORNEK["son_7_gun"]])).json()["gunluk_toplam"]
    assert cok > 1.5 * az


def test_eksik_saat(istemci):
    assert istemci.post("/tahmin", json=istek(son_7_gun=ORNEK["son_7_gun"][:100])).status_code == 422


def test_negatif_tuketim(istemci):
    assert istemci.post("/tahmin", json=istek(son_7_gun=[-1.0] * 168)).status_code == 422


def test_bilinmeyen_musteri(istemci):
    assert istemci.post("/tahmin", json=istek(musteri="MT_999")).status_code == 404
