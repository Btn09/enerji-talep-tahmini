"""Gün öncesi tahmin için çok katmanlı sinir ağı: son 7 günün 168 saatinden ertesi günün 24 saatini birlikte tahmin eder."""
import numpy as np
import torch
from torch import nn


class GunOncesiAg(nn.Module):
    def __init__(self, girdi=180, gizli=512, katman=3, cikti=24, birakma=0.1):
        super().__init__()
        bloklar, boyut = [], girdi
        for _ in range(katman):
            bloklar += [nn.Linear(boyut, gizli), nn.ReLU(), nn.Dropout(birakma)]
            boyut = gizli
        self.govde = nn.Sequential(*bloklar)
        self.cikis = nn.Linear(gizli, cikti)

    def forward(self, x):
        # Kalıntı bağlantı: ağ, dünün profilinden farkı öğreniyor
        dun = x[:, 144:168]
        return dun + self.cikis(self.govde(x))


def egit(X_e, y_e, X_d, y_d, tur=40, grup=1024, ogrenme=1e-3, cihaz=None, sabir=6):
    cihaz = cihaz or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = GunOncesiAg(X_e.shape[1]).to(cihaz)
    opt = torch.optim.AdamW(model.parameters(), lr=ogrenme, weight_decay=1e-4)
    zam = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tur)
    Xe, ye = torch.from_numpy(X_e).to(cihaz), torch.from_numpy(y_e).to(cihaz)
    Xd, yd = torch.from_numpy(X_d).to(cihaz), torch.from_numpy(y_d).to(cihaz)
    en_iyi, en_iyi_durum, bekleme, gecmis = np.inf, None, 0, []
    for _ in range(tur):
        model.train()
        for sira in torch.randperm(len(Xe), device=cihaz).split(grup):
            # Mutlak hata: büyük ani sıçramalara karşı daha dayanıklı
            kayip = (model(Xe[sira]) - ye[sira]).abs().mean()
            opt.zero_grad()
            kayip.backward()
            opt.step()
        zam.step()
        model.eval()
        with torch.no_grad():
            d = (model(Xd) - yd).abs().mean().item()
        gecmis.append(d)
        if d < en_iyi:
            en_iyi, en_iyi_durum, bekleme = d, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bekleme += 1
            if bekleme >= sabir:
                break
    model.load_state_dict(en_iyi_durum)
    return model.eval(), gecmis


@torch.no_grad()
def tahmin(model, X, grup=65536):
    cihaz = next(model.parameters()).device
    return np.concatenate([model(torch.from_numpy(X[i:i + grup]).to(cihaz)).cpu().numpy()
                           for i in range(0, len(X), grup)])
