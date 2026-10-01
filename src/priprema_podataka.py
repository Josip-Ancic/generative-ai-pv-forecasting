from pathlib import Path

import numpy as np
import pandas as pd

KUCANSTVO = "residential4"
POCETAK = "2015-10-11"
KRAJ = "2018-02-03"
NAJMANJE_SATI = 22
FAKTOR_STRSILA = 3

MAPA = Path(__file__).resolve().parent.parent
ULAZNI_CSV = MAPA / "household_data_60min_singleindex.csv"
MAPA_PODACI = MAPA / "podaci"


def ucitaj_sirove_podatke():
    return pd.read_csv(ULAZNI_CSV, parse_dates=["utc_timestamp"],
                       index_col="utc_timestamp", low_memory=False)


def satne_vrijednosti(sirovi_podaci):
    proizvodnja = sirovi_podaci[f"DE_KN_{KUCANSTVO}_pv"]
    preuzeto = sirovi_podaci[f"DE_KN_{KUCANSTVO}_grid_import"]
    predano = sirovi_podaci[f"DE_KN_{KUCANSTVO}_grid_export"]

    satno = pd.DataFrame({"pv": proizvodnja.diff(),
                          "imp": preuzeto.diff(),
                          "exp": predano.diff()})
    satno[satno < 0] = np.nan
    for stupac in satno:
        granica = satno[stupac].quantile(0.9999) * FAKTOR_STRSILA
        satno.loc[satno[stupac] > granica, stupac] = np.nan

    satno["cons"] = satno["pv"] + satno["imp"] - satno["exp"]
    satno.loc[satno["cons"] < 0, "cons"] = np.nan
    satno = satno.dropna(subset=["pv", "cons"])
    return satno[POCETAK:KRAJ]


def potpuni_dani(satno):
    dnevno = satno.resample("D").agg({"pv": "sum", "cons": "sum",
                                      "imp": "sum", "exp": "sum"})
    dnevno["izmjereno_sati"] = satno["pv"].resample("D").count()
    dnevno = dnevno[dnevno["izmjereno_sati"] >= NAJMANJE_SATI]
    return dnevno.drop(columns="izmjereno_sati")


def main():
    sirovi_podaci = ucitaj_sirove_podatke()
    satno = satne_vrijednosti(sirovi_podaci)
    dnevno = potpuni_dani(satno)

    MAPA_PODACI.mkdir(exist_ok=True)
    satno.to_csv(MAPA_PODACI / f"{KUCANSTVO}_satno.csv")
    dnevno.to_csv(MAPA_PODACI / f"{KUCANSTVO}_dnevno.csv")

    print(f"satnih zapisa: {len(satno)} | potpunih dana: {len(dnevno)}")
    print(dnevno.describe().round(2))


if __name__ == "__main__":
    main()
