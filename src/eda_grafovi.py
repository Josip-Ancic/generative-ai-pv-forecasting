import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stil_grafova import (MAPA_PODACI, ZELENA, NARANCASTA, postavi_stil, spremi)

SEZONE = {"Zima (12–2)": [12, 1, 2],
          "Proljeće (3–5)": [3, 4, 5],
          "Ljeto (6–8)": [6, 7, 8],
          "Jesen (9–11)": [9, 10, 11]}


def ucitaj(naziv):
    return pd.read_csv(MAPA_PODACI / naziv, parse_dates=["utc_timestamp"],
                       index_col="utc_timestamp")


def graf_mjesecnih_vrijednosti(dnevno):
    mjesecno = dnevno.resample("MS").sum()["2015-11-01":"2018-01-01"]
    figura, os = plt.subplots(figsize=(7.2, 3.2), dpi=150)
    polozaj = np.arange(len(mjesecno))
    sirina = 0.4
    os.bar(polozaj - sirina / 2, mjesecno["pv"], sirina,
           label="Proizvodnja PV-a", color=ZELENA)
    os.bar(polozaj + sirina / 2, mjesecno["cons"], sirina,
           label="Potrošnja", color=NARANCASTA)
    oznake = [f"{dan.month}/{dan.year % 100}" for dan in mjesecno.index]
    os.set_xticks(polozaj[::2])
    os.set_xticklabels(oznake[::2])
    os.set_ylabel("Energija (kWh)")
    os.legend()
    spremi(figura, "slika_mjesecna_proizvodnja_potrosnja.png")


def graf_dnevnih_profila(satno):
    figura, osi = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=150, sharex=True)
    for naziv_sezone, mjeseci in SEZONE.items():
        sezona = satno[satno.index.month.isin(mjeseci)]
        po_satu = sezona.groupby(sezona.index.hour)
        osi[0].plot(po_satu["pv"].mean(), label=naziv_sezone)
        osi[1].plot(po_satu["cons"].mean(), label=naziv_sezone)
    osi[0].set_title("Proizvodnja PV-a")
    osi[1].set_title("Potrošnja")
    for os in osi:
        os.set_xlabel("Sat u danu (UTC)")
        os.set_xticks(range(0, 24, 4))
    osi[0].set_ylabel("Prosječna energija (kWh/h)")
    osi[0].legend(fontsize=7)
    spremi(figura, "slika_dnevni_profili_sezone.png")


def main():
    postavi_stil()
    satno = ucitaj("residential4_satno.csv")
    dnevno = ucitaj("residential4_dnevno.csv")
    graf_mjesecnih_vrijednosti(dnevno)
    graf_dnevnih_profila(satno)
    print("Grafovi eksplorativne analize spremljeni.")


if __name__ == "__main__":
    main()
