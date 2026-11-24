"""Gün öncesi elektrik talebi tahmin servisi.

Çalıştırma:  uvicorn api.app:app
Dokümantasyon: http://127.0.0.1:8000/docs
"""
import datetime as dt
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.cikarim import Tahminci


class Istek(BaseModel):
    musteri: str = Field(examples=["MT_001"])
    hedef_tarih: dt.date
    son_7_gun: Annotated[list[Annotated[float, Field(ge=0)]], Field(min_length=168, max_length=168,
                         description="Hedef günden önceki 7 günün saatlik tüketimi (kWh), eskiden yeniye")]


class Saat(BaseModel):
    saat: int
    tahmin: float
    alt: float
    ust: float


class Cevap(BaseModel):
    musteri: str
    hedef_tarih: dt.date
    gunluk_toplam: float
    saatler: list[Saat]


durum = {}


@asynccontextmanager
async def lifespan(app):
    durum["t"] = Tahminci()
    yield
    durum.clear()


app = FastAPI(title="Elektrik Talebi Tahmini", lifespan=lifespan)


@app.get("/musteriler")
def musteriler():
    return durum["t"].musteriler


@app.post("/tahmin", response_model=Cevap)
def tahmin(istek: Istek):
    """Ertesi günün 24 saati için tahmin ve %80 tahmin aralığı (alt: 10. yüzdelik, üst: 90. yüzdelik)."""
    if istek.musteri not in durum["t"].sira:
        raise HTTPException(404, f"Bilinmeyen müşteri: {istek.musteri}")
    orta, alt, ust = durum["t"].tahmin_et(istek.musteri, istek.hedef_tarih, istek.son_7_gun)
    return {"musteri": istek.musteri, "hedef_tarih": istek.hedef_tarih, "gunluk_toplam": round(float(orta.sum()), 2),
            "saatler": [{"saat": h, "tahmin": round(float(o), 3), "alt": round(float(a), 3), "ust": round(float(u), 3)}
                        for h, (o, a, u) in enumerate(zip(orta, alt, ust))]}
