# Generative AI for Household Solar (PV) Forecasting: Conditional VAE

**Bachelor's thesis**: *Generative Artificial Intelligence in Renewable Energy Systems*
Faculty of Organization and Informatics (FOI), University of Zagreb, September 2026 · Mentor: Assoc. Prof. Dijana Oreški, PhD

A **conditional variational autoencoder (cVAE), written from scratch in NumPy**, learns the daily profiles of solar production and household consumption from real hourly smart-meter data. Instead of predicting a single number for tomorrow, it generates hundreds of plausible scenarios, from cloudy to fully sunny days, so the forecast also describes uncertainty.

## Results (seasonally balanced test set)

| Series | Model | MAE (kWh/h) | RMSE (kWh/h) |
|---|---|---|---|
| PV production | **cVAE** | **0.515** | **1.021** |
| PV production | Climatology | 0.542 | 1.038 |
| PV production | Persistence | 0.560 | 1.296 |
| Consumption | **cVAE** | **0.347** | **0.540** |
| Consumption | Climatology | 0.348 | 0.544 |
| Consumption | Persistence | 0.401 | 0.699 |

- The cVAE beats both baselines on both series. Against persistence, RMSE is more than 20 % lower.
- Generated scenarios follow the real seasonal patterns. In June the distribution of daily production matches the real one, including weaker and very sunny days.

## How it works

1. **Data prep** (`priprema_podataka.py`): cumulative meter readings are turned into hourly energy with `diff()`, impossible jumps and gaps are removed, and consumption is computed as `PV + import − export`.
2. **EDA** (`eda_grafovi.py`): seasonality (summer production up to 5× higher) and the shape of a typical day.
3. **Model** (`cvae_model.py`): encoder and decoder with a latent dimension of 8, conditioned on the month, the previous day's total energy and a weekend flag, plus KL warm-up (β-annealing) and mini-batch training. The script trains, forecasts, compares against climatology and persistence, and saves the scenarios.
4. **Figures** (`rezultati_grafovi.py`, `stil_grafova.py`).
5. **Notebook** (`analiza_i_model.ipynb`): the whole pipeline in one place, with outputs.

## Data

[Open Power System Data, Household Data](https://data.open-power-system-data.org/household_data/) (60-min, single index), household `DE_KN_residential4`, 2015-10 to 2018-02.
Download `household_data_60min_singleindex.csv` into the repository root (it is not included because of its size).

## Run

```bash
pip install numpy pandas matplotlib
python src/priprema_podataka.py   # -> podaci/residential4_satno.csv
python src/cvae_model.py          # train + evaluate + scenarios
python src/eda_grafovi.py
python src/rezultati_grafovi.py
```

## Tech stack

Python · NumPy (neural network written from scratch, no deep learning framework) · pandas · Matplotlib · Jupyter

The full thesis (in Croatian) is in [`docs/`](docs/).

---

## Hrvatski

# Generativna umjetna inteligencija u sustavima obnovljivih izvora energije

**Završni rad**, FOI Varaždin, rujan 2026. · Mentorica: izv. prof. dr. sc. Dijana Oreški

**Uvjetni varijacijski autoenkoder (cVAE), napisan od nule u NumPyju**, uči dnevne profile proizvodnje fotonaponskog sustava i potrošnje kućanstva iz stvarnih satnih mjerenja. Umjesto jedne prognoze za sutra generira stotine mogućih scenarija, pa prognoza opisuje i neizvjesnost.

### Rezultati

cVAE ima najmanju pogrešku i za proizvodnju (MAE 0,515 i RMSE 1,021 kWh/h) i za potrošnju (MAE 0,347 i RMSE 0,540 kWh/h). Bolji je od klimatologije i od persistencije, a od persistencije je bolji za više od 20 % po RMSE-u. Generirani scenariji vjerno prate sezonske obrasce.

### Struktura

- `src/priprema_podataka.py`: čišćenje i satne vrijednosti
- `src/eda_grafovi.py`: eksplorativna analiza
- `src/cvae_model.py`: model, treniranje, usporedba s referentnim modelima, scenariji
- `src/rezultati_grafovi.py`: grafovi rezultata
- `src/analiza_i_model.ipynb`: cijeli postupak u jednom notebooku
- `docs/`: tekst završnog rada

### Podaci

Open Power System Data, *Household Data*. Datoteku `household_data_60min_singleindex.csv` preuzmi u korijen repozitorija pa pokreni skripte redom (naredbe su iznad).
