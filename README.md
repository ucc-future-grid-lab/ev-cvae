# Synthetic EV Charging Session Generation Using a Conditional Variational Autoencoder

## Overview

This work proposes a Conditional Variational Autoencoder (CVAE) for generating synthetic electric vehicle (EV) charging sessions from real transaction-level data. The model is trained on session-level features describing plug-in duration, charging duration, delivered energy, charging delay, and cyclical time-of-week, while conditioning on day-of-week and managed charging status.

The generated synthetic data is evaluated using distributional similarity metrics (Wasserstein distance, Spearman correlation MAE, ACF MAE) and utility-based validation via the Train-on-Synthetic-Test-on-Real (TSTR) protocol.

## Repository Structure

```
|-- data/raw/
    |-- CrowdCharge_Transactions.parquet                # Raw CrowdCharge session data (Electric Nation)
    |-- GreenFlux_Transactions.parquet                  # Raw GreenFlux session data (Electric Nation)
|-- notebooks/
    |-- EV_charging_transactions_preprocessing.ipynb    # Data preprocessing pipeline
    |-- EV_CVAE_train_eval.ipynb                        # CVAE training and evaluation
|-- src/ev_cvae/
    |-- ev_cvae.py                                      # CVAE class using PyTorch
    |-- transform.py                                    # Data transformation and scaling functions
    |-- metrics.py                                      # Evaluation metrics functions
|-- README.md
```

## Requirements

This code is designed to work with `python>=3.13`. Please make sure `python` is installed on your system by running the following command on a terminal (GNU/Linux or macOS) or on the Command Prompt/PowerShell (Windows)

```
python --version
```

### Dependencies

For convenience, all dependencies could be installed using the [uv](https://docs.astral.sh/uv/) package manager. Please refer to the [Official Installation Instructions](https://docs.astral.sh/uv/getting-started/installation/) for your OS.

### Local environment installation

Once `uv` is installed, the full environment (including optional dependencies) could be set up by running the following command from the main project folder.

```
uv sync --all-groups --all-extras
```

### Data

The raw data is sourced from the [Electric Nation UK](https://www.electricnation.org.uk/) smart charging trial (CrowdCharge and GreenFlux datasets), covering residential EV charging sessions from early 2017 to end of 2018 across 602 charging points (\~157k sessions combined).

For convenience, both datasets are provided in the repository as `.parquet` in the `data/raw` folder.

## Notebooks

The main workflow is provided in two separate Jupyter Notebooks.

### `EV_charging_transactions_preprocessing.ipynb`

Merges the CrowdCharge and GreenFlux datasets, filters outliers, selects and engineers features, applies cyclic time encoding, and saves the preprocessed dataset ready for model training.

### `EV_CVAE_train_eval.ipynb`

Defines and trains the CVAE model, produces training diagnostics, evaluates generated data against real data using distributional and utility-based metrics, and generates synthetic charging sessions conditioned on day-of-week and managed/unmanaged status.

## Model

### Engineered Features

| Feature                 | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| `PluggedInTime`         | Total plug-in duration (hours)                                |
| `ChargingDuration`      | Active charging duration (hours)                              |
| `ConsumedkWh`           | Energy delivered during the session (kWh)                     |
| `TimeGapHours`          | Time between plug-in and charging start (hours)               |
| `WeekTime_sin`          | Sine component of cyclical week-time encoding                 |
| `WeekTime_cos`          | Cosine component of cyclical week-time encoding               |
| `DayOfWeek`             | Day of the week (0–6, one-hot encoded, conditioning variable) |
| `Part_of_Managed_Group` | Smart charging group membership (0/1, conditioning variable)  |

### Architecture

The CVAE uses a fully connected MLP architecture for both encoder and decoder:

- **Input dimension:** 14 (6 continuous features + 8-dimensional condition vector)
- **Hidden layers:** 2 × 256 neurons with ReLU activations
- **Latent dimension:** 64
- **Reconstruction loss:** Gaussian negative log-likelihood (NLL)
- **Regularisation:** KL divergence toward standard normal prior (β = 0.5)
- **Optimiser:** Adam, lr = 3×10⁻⁴, 60 epochs, batch size 256

## Citation

If you use this code, please cite:

```tex
@InProceedings{Maye2025,
  author    = {Kelly, Graeme and Palacios-Garcia, Emilio J. and Hayes, Barry P.},
  booktitle = {2026 {{IEEE PES Innovative Smart Grid Technologies Europe}} ({{ISGT EUROPE}})},
  date      = {2026-10},
  title     = {{Synthetic Electric Vehicle Charging Session Generation Using a Conditional Variational Autoencoder}},
  pages     = {1--5},
}
```

## Licence

This project is licensed under the [MIT License](LICENSE)

## Acknowledgments

This publication has emanated from research conducted with the financial support of Taighde Éireann – Research Ireland under Grant numbers 12/RC/2302_P2 and 22/FFP-A/10455.
