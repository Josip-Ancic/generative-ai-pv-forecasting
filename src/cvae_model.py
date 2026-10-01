from pathlib import Path

import numpy as np
import pandas as pd


BROJ_SATI = 24
BROJ_UVJETA = 14
LATENTNE_DIM = 8
SKRIVENI_SLOJ = 64
BROJ_EPOHA = 3000
VELICINA_MINIGRUPE = 64
STOPA_UCENJA = 1e-3
BETA_MAX = 0.1
EPOHA_ZAGRIJAVANJA = 500


UDIO_ZA_TESTIRANJE = 0.25
BROJ_SCENARIJA_PROGNOZA = 200
BROJ_SCENARIJA_PRIKAZ = 500
SJEME = 42
SJEME_TEZINA = 1

MAPA = Path(__file__).resolve().parent.parent
ULAZNI_CSV = MAPA / "podaci" / "residential4_satno.csv"
MAPA_PODACI = MAPA / "podaci"


def ucitaj_satne_podatke():
    return pd.read_csv(ULAZNI_CSV, parse_dates=["utc_timestamp"],
                       index_col="utc_timestamp")


def dnevni_profili(satni_podaci, serija):
    profili = satni_podaci.pivot_table(index=satni_podaci.index.date,
                                       columns=satni_podaci.index.hour,
                                       values=serija).dropna()
    profili.index = pd.to_datetime(profili.index)
    return profili


def pripremi_ulaze(satni_podaci, serija):
    profili = dnevni_profili(satni_podaci, serija)
    energija_prethodnog_dana = profili.sum(axis=1).shift(1)
    ima_prethodni = energija_prethodnog_dana.notna()

    X = profili[ima_prethodni].values
    datumi = profili.index[ima_prethodni]
    mjesec = np.eye(12)[datumi.month - 1]
    prethodni_dan = energija_prethodnog_dana[ima_prethodni].values[:, None]
    vikend = (datumi.dayofweek >= 5).astype(float)[:, None]
    return X, datumi, mjesec, prethodni_dan, vikend


def podijeli_dane(datumi, rng):
    je_testni = np.zeros(len(datumi), dtype=bool)
    for mjesec in range(1, 13):
        indeksi = np.where(datumi.month == mjesec)[0]
        if len(indeksi) == 0:
            continue
        koliko = int(round(len(indeksi) * UDIO_ZA_TESTIRANJE))
        je_testni[rng.choice(indeksi, koliko, replace=False)] = True
    return ~je_testni


class CVAE:

    def __init__(self, broj_ulaza, broj_uvjeta, rng,
                 latentne_dim=LATENTNE_DIM, skriveni_sloj=SKRIVENI_SLOJ,
                 sjeme_tezina=SJEME_TEZINA):
        pocetni = np.random.default_rng(sjeme_tezina)

        def tezine(ulaz, izlaz):
            return pocetni.normal(0, np.sqrt(2 / (ulaz + izlaz)), (ulaz, izlaz))

        self.rng = rng
        self.latentne_dim = latentne_dim


        self.W1 = tezine(broj_ulaza + broj_uvjeta, skriveni_sloj)
        self.b1 = np.zeros(skriveni_sloj)
        self.W_sredina = tezine(skriveni_sloj, latentne_dim)
        self.b_sredina = np.zeros(latentne_dim)
        self.W_logvar = tezine(skriveni_sloj, latentne_dim)
        self.b_logvar = np.zeros(latentne_dim)


        self.W2 = tezine(latentne_dim + broj_uvjeta, skriveni_sloj)
        self.b2 = np.zeros(skriveni_sloj)
        self.W3 = tezine(skriveni_sloj, broj_ulaza)
        self.b3 = np.zeros(broj_ulaza)

        self.parametri = ["W1", "b1", "W_sredina", "b_sredina", "W_logvar",
                          "b_logvar", "W2", "b2", "W3", "b3"]

        self.prvi_moment = {ime: 0 for ime in self.parametri}
        self.drugi_moment = {ime: 0 for ime in self.parametri}
        self.korak = 0

    def dekodiraj(self, latentni, uvjeti):
        skriveni = np.tanh(np.hstack([latentni, uvjeti]) @ self.W2 + self.b2)
        return skriveni @ self.W3 + self.b3

    def korak_treniranja(self, X, uvjeti, beta, stopa_ucenja=STOPA_UCENJA):
        velicina = len(X)


        ulaz_enkodera = np.hstack([X, uvjeti])
        skriveni_enk = np.tanh(ulaz_enkodera @ self.W1 + self.b1)
        sredina = skriveni_enk @ self.W_sredina + self.b_sredina
        log_varijanca = np.clip(skriveni_enk @ self.W_logvar + self.b_logvar, -8, 8)


        sum_ = self.rng.normal(size=sredina.shape)
        latentni = sredina + sum_ * np.exp(0.5 * log_varijanca)


        ulaz_dekodera = np.hstack([latentni, uvjeti])
        skriveni_dek = np.tanh(ulaz_dekodera @ self.W2 + self.b2)
        rekonstrukcija = skriveni_dek @ self.W3 + self.b3


        d_rekonstrukcija = 2 * (rekonstrukcija - X) / velicina
        g_W3 = skriveni_dek.T @ d_rekonstrukcija
        g_b3 = d_rekonstrukcija.sum(0)
        d_skriveni_dek = (d_rekonstrukcija @ self.W3.T) * (1 - skriveni_dek ** 2)
        g_W2 = ulaz_dekodera.T @ d_skriveni_dek
        g_b2 = d_skriveni_dek.sum(0)


        d_latentni = (d_skriveni_dek @ self.W2.T)[:, :self.latentne_dim]
        d_sredina = d_latentni + beta * sredina / velicina
        d_log_varijanca = (d_latentni * sum_ * 0.5 * np.exp(0.5 * log_varijanca)
                           + beta * 0.5 * (np.exp(log_varijanca) - 1) / velicina)


        g_W_sredina = skriveni_enk.T @ d_sredina
        g_b_sredina = d_sredina.sum(0)
        g_W_logvar = skriveni_enk.T @ d_log_varijanca
        g_b_logvar = d_log_varijanca.sum(0)
        d_skriveni_enk = ((d_sredina @ self.W_sredina.T
                           + d_log_varijanca @ self.W_logvar.T)
                          * (1 - skriveni_enk ** 2))
        g_W1 = ulaz_enkodera.T @ d_skriveni_enk
        g_b1 = d_skriveni_enk.sum(0)

        gradijenti = {"W1": g_W1, "b1": g_b1,
                      "W_sredina": g_W_sredina, "b_sredina": g_b_sredina,
                      "W_logvar": g_W_logvar, "b_logvar": g_b_logvar,
                      "W2": g_W2, "b2": g_b2, "W3": g_W3, "b3": g_b3}
        self._azuriraj_adamom(gradijenti, stopa_ucenja)

    def _azuriraj_adamom(self, gradijenti, stopa_ucenja):
        self.korak += 1
        for ime in self.parametri:
            gradijent = gradijenti[ime]
            self.prvi_moment[ime] = 0.9 * self.prvi_moment[ime] + 0.1 * gradijent
            self.drugi_moment[ime] = (0.999 * self.drugi_moment[ime]
                                      + 0.001 * gradijent ** 2)
            prvi = self.prvi_moment[ime] / (1 - 0.9 ** self.korak)
            drugi = self.drugi_moment[ime] / (1 - 0.999 ** self.korak)
            nova = getattr(self, ime) - stopa_ucenja * prvi / (np.sqrt(drugi) + 1e-8)
            setattr(self, ime, nova)


def treniraj(model, X, uvjeti, rng):
    indeksi = np.arange(len(X))
    for epoha in range(BROJ_EPOHA):
        beta = min(BETA_MAX, BETA_MAX * epoha / EPOHA_ZAGRIJAVANJA)
        rng.shuffle(indeksi)
        for pocetak in range(0, len(indeksi), VELICINA_MINIGRUPE):
            minigrupa = indeksi[pocetak:pocetak + VELICINA_MINIGRUPE]
            model.korak_treniranja(X[minigrupa], uvjeti[minigrupa], beta)
    return model


def prognoziraj(model, uvjeti, mjerilo):
    prognoze = []
    for i in range(len(uvjeti)):
        sum_ = model.rng.normal(size=(BROJ_SCENARIJA_PROGNOZA, model.latentne_dim))
        uvjeti_dana = np.repeat(uvjeti[i:i + 1], BROJ_SCENARIJA_PROGNOZA, axis=0)
        scenariji = np.clip(model.dekodiraj(sum_, uvjeti_dana), 0, None)
        prognoze.append(scenariji.mean(axis=0) * mjerilo)
    return np.array(prognoze)


def klimatologija(profili, datumi_treniranja):
    skup = set(datumi_treniranja)
    prosjeci = {}
    for mjesec in range(1, 13):
        dani = [dan for dan in profili.index
                if dan.month == mjesec and dan in skup]
        prosjeci[mjesec] = profili.loc[dani].values.mean(axis=0)
    return prosjeci


def persistencija(profili, testni_datumi, prosjeci_mjeseci):
    prognoze = []
    for dan in testni_datumi:
        prethodni = profili.loc[profili.index == dan - pd.Timedelta(days=1)]
        if len(prethodni):
            prognoze.append(prethodni.values[0])
        else:
            prognoze.append(prosjeci_mjeseci[dan.month])
    return np.array(prognoze)


def mjere_pogreske(prognoza, stvarno):
    mae = np.abs(prognoza - stvarno).mean()
    rmse = np.sqrt(((prognoza - stvarno) ** 2).mean())
    return mae, rmse


def spremi_scenarije(model, ime_serije, mjesec, oznaka, uvjeti_izvor, mjerilo):
    prethodni_dani, vikendi = uvjeti_izvor
    odabrani = model.rng.integers(0, len(prethodni_dani), BROJ_SCENARIJA_PRIKAZ)
    uvjeti = np.zeros((BROJ_SCENARIJA_PRIKAZ, BROJ_UVJETA))
    uvjeti[:, mjesec - 1] = 1
    uvjeti[:, 12] = prethodni_dani[odabrani]
    uvjeti[:, 13] = vikendi[odabrani]

    sum_ = model.rng.normal(size=(BROJ_SCENARIJA_PRIKAZ, model.latentne_dim))
    scenariji = np.clip(model.dekodiraj(sum_, uvjeti), 0, None) * mjerilo
    np.save(MAPA_PODACI / f"scenariji_{ime_serije}_{oznaka}.npy", scenariji)
    return scenariji


def obradi_seriju(satni_podaci, stupac, ime_serije):


    rng_podjele = np.random.default_rng(SJEME)
    rng = np.random.default_rng(SJEME)

    X, datumi, mjesec, prethodni_dan, vikend = pripremi_ulaze(satni_podaci, stupac)
    za_treniranje = podijeli_dane(datumi, rng_podjele)


    mjerilo = X[za_treniranje].max()
    mjerilo_prethodnog = prethodni_dan[za_treniranje].max()
    X_norm = X / mjerilo
    uvjeti = np.hstack([mjesec, prethodni_dan / mjerilo_prethodnog, vikend])

    testni_datumi = datumi[~za_treniranje]
    X_testni = X[~za_treniranje]

    model = CVAE(BROJ_SATI, BROJ_UVJETA, rng)
    treniraj(model, X_norm[za_treniranje], uvjeti[za_treniranje], rng)

    prognoza_modela = prognoziraj(model, uvjeti[~za_treniranje], mjerilo)

    profili = dnevni_profili(satni_podaci, stupac)
    prosjeci_mjeseci = klimatologija(profili, datumi[za_treniranje])
    prognoza_klimatologije = np.array([prosjeci_mjeseci[dan.month]
                                       for dan in testni_datumi])
    prognoza_persistencije = persistencija(profili, testni_datumi, prosjeci_mjeseci)

    print(f"== {ime_serije} ==")
    for naziv, prognoza in [("cVAE", prognoza_modela),
                            ("Klimatologija", prognoza_klimatologije),
                            ("Persistencija", prognoza_persistencije)]:
        mae, rmse = mjere_pogreske(prognoza, X_testni)
        print(f"  {naziv:<14} MAE={mae:.3f} RMSE={rmse:.3f} kWh/h")

    np.savez(MAPA_PODACI / f"rezultati_{ime_serije}.npz",
             preds=prognoza_modela, clim=prognoza_klimatologije,
             pers=prognoza_persistencije, real=X_testni,
             dates=testni_datumi.astype(str))

    for broj_mjeseca, oznaka in [(1, "sijecanj"), (6, "lipanj")]:
        maska = za_treniranje & (datumi.month == broj_mjeseca)
        uvjeti_izvor = (prethodni_dan[maska][:, 0] / mjerilo_prethodnog,
                        vikend[maska][:, 0])
        spremi_scenarije(model, ime_serije, broj_mjeseca, oznaka,
                         uvjeti_izvor, mjerilo)


def main():
    satni_podaci = ucitaj_satne_podatke()
    for stupac, ime_serije in [("pv", "pv"), ("cons", "potrosnja")]:
        obradi_seriju(satni_podaci, stupac, ime_serije)
    print("Gotovo.")


if __name__ == "__main__":
    main()
