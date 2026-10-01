import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stil_grafova import (MAPA_PODACI, ZELENA, TAMNOZELENA, SIVA,
                          postavi_stil, spremi)

BROJ_PRIKAZANIH_SCENARIJA = 60
DAN_PROIZVODNJE = 7
DAN_POTROSNJE = 11


def ucitaj_satne_podatke():
    return pd.read_csv(MAPA_PODACI / "residential4_satno.csv",
                       parse_dates=["utc_timestamp"], index_col="utc_timestamp")


def graf_scenarija(satno):
    figura, osi = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=150, sharey=True)
    mjeseci = [("sijecanj", 1, "Siječanj"), ("lipanj", 6, "Lipanj")]
    for os, (oznaka, broj_mjeseca, naslov) in zip(osi, mjeseci):
        scenariji = np.load(MAPA_PODACI / f"scenariji_pv_{oznaka}.npy")
        for scenarij in scenariji[:BROJ_PRIKAZANIH_SCENARIJA]:
            os.plot(scenarij, color=ZELENA, alpha=0.12, lw=0.7)
        os.plot(scenariji.mean(axis=0), color=TAMNOZELENA, lw=2,
                label="Prosjek scenarija (cVAE)")
        stvarni = satno[satno.index.month == broj_mjeseca]
        prosjek_po_satu = stvarni.groupby(stvarni.index.hour)["pv"].mean()
        os.plot(prosjek_po_satu.values, "k--", lw=1.8, label="Stvarni prosjek")
        os.set_title(naslov)
        os.set_xlabel("Sat u danu (UTC)")
    osi[0].set_ylabel("Proizvodnja (kWh/h)")
    osi[0].legend(fontsize=7)
    spremi(figura, "slika_scenariji_pv.png")


def graf_razdiobe(satno):
    scenariji = np.load(MAPA_PODACI / "scenariji_pv_lipanj.npy")
    lipanj = satno[satno.index.month == 6]["pv"].resample("D").sum()
    lipanj = lipanj[lipanj > 0]

    figura, os = plt.subplots(figsize=(4.8, 2.8), dpi=150)
    razredi = np.linspace(10, 70, 20)
    os.hist(lipanj.values, bins=razredi, density=True, alpha=0.55, color="k",
            label="Stvarni dani (lipanj)")
    os.hist(scenariji.sum(axis=1), bins=razredi, density=True, alpha=0.55,
            color=ZELENA, label="cVAE scenariji")
    os.set_xlabel("Dnevna proizvodnja (kWh)")
    os.set_ylabel("Gustoća")
    os.legend(fontsize=7)
    spremi(figura, "slika_distribucija_lipanj.png")


def odaberi_dan(rezultati, mjesec):
    datumi = pd.to_datetime(rezultati["dates"])
    for redni_broj, datum in enumerate(datumi):
        if datum.month == mjesec:
            return redni_broj, datum
    raise ValueError(f"U testnom skupu nema dana u mjesecu {mjesec}.")


def graf_prognoza():
    proizvodnja = np.load(MAPA_PODACI / "rezultati_pv.npz", allow_pickle=True)
    potrosnja = np.load(MAPA_PODACI / "rezultati_potrosnja.npz", allow_pickle=True)
    indeks_pv, datum_pv = odaberi_dan(proizvodnja, DAN_PROIZVODNJE)
    indeks_pot, datum_pot = odaberi_dan(potrosnja, DAN_POTROSNJE)

    figura, osi = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=150)
    for os, rezultati, indeks, naslov in [
            (osi[0], proizvodnja, indeks_pv,
             f"Proizvodnja PV-a, {datum_pv:%d.%m.%Y.}"),
            (osi[1], potrosnja, indeks_pot,
             f"Potrošnja, {datum_pot:%d.%m.%Y.}")]:
        os.plot(rezultati["real"][indeks], "k-", lw=1.6, label="Stvarno")
        os.plot(rezultati["preds"][indeks], color=TAMNOZELENA, lw=1.6, label="cVAE")
        os.plot(rezultati["clim"][indeks], color=SIVA, ls=":", lw=1.4,
                label="Klimatologija")
        os.set_title(naslov)
        os.set_xlabel("Sat u danu (UTC)")
    osi[0].set_ylabel("kWh/h")
    osi[0].legend(fontsize=7)
    spremi(figura, "slika_prognoza_primjeri.png")


def main():
    postavi_stil()
    satno = ucitaj_satne_podatke()
    graf_scenarija(satno)
    graf_razdiobe(satno)
    graf_prognoza()
    print("Grafovi rezultata spremljeni.")


if __name__ == "__main__":
    main()
