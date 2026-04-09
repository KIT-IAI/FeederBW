# FeederBW

This is a companion repository for exploring the **FeederBW dataset** — two years of 1-minute resolution measurements from 200 low-voltage distribution feeders in Baden-Württemberg, Germany.

📄 **Preprint:** [arXiv:2602.03521](https://arxiv.org/abs/2602.03521)  
📦 **Dataset:** [Zenodo Record 17831177](https://zenodo.org/records/17831177)

## Dataset Overview

| Source | Resolution | Description |
|---|---|---|
| **Measurement data** | 1 min | Active/reactive/apparent power, voltage, per-phase currents & power factors for 200 feeders (April 2023 – March 2025) |
| **Weather data** | 1 h | Solar radiation, temperature, precipitation, humidity, wind, snow height — per feeder location |
| **Metadata** | Daily | Number of grid connection customers, installed power (PV, batteries, EV chargers, heat pumps), average consumption of industry |

Measurement data is split into 5 batches of 40 feeders each for better data handling and can be downloaded selectively. The size of all data unzipped is around 12.1 GB (zipped 8.6 GB). 

## Getting Started

### Prerequisites

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone https://github.com/KIT-IAI/FeederBW.git
cd FeederBW
uv sync
```

### Quick Start

Run the pipeline with reduced data (download → load → forecast → pseudo-measurement generation):

```bash
uv run python main.py
```

Or explore interactively via the [tutorial.ipynb](tutorial.ipynb) notebook.

## Usage

### Download Data

```python
from src.download import download_data

# Download everything
download_data()

# Download specific subsets
download_data(files=["feeder_metadata", "weather_data", "001-040"])
```

**Available file identifiers:** `"feeder_metadata"`, `"weather_data"`, `"001-040"`, `"041-080"`, `"081-120"`, `"121-160"`, `"161-200"`

### Load Data

```python
from src.load import load_data

# Load specific feeders, columns, and time range at 15-min resolution
df = load_data(
    feeders=[1, 2, 3],
    columns=["active_power_kW", "air_temperature_C", "PV_systems_kW"],
    time_period=("2024-01-01", "2024-04-01"),
    temporal_resolution="15min",
    aggregation_type={"active_power_kW": "mean", "air_temperature_C": "mean"},
)
```

Measurement, weather, and metadata columns are automatically joined based on the requested columns.

### Forecast Baseline

Naive quantile forecast using historical same-time-of-day values without additional features:

```python
from src.model.forecast import forecast

metrics = forecast(data=df, split_date="2024-03-01", target_column="active_power_kW")
```

### Pseudo-Measurement Generation

Cross-feeder quantile regression using `HistGradientBoostingRegressor` based on a few feeder metadata columns:

```python
from src.model.pseudo_measurement_generation import pseudo_measurement_generation

metrics = pseudo_measurement_generation(
    data=df,
    target_column="active_power_kW",
    feature_columns=["housing_units_count", "PV_systems_kW"],
    test_feeders=[3],
)
```

Both models output per-feeder metrics (MAE, pinball loss, Winkler score) as CSV and visualizations of nine random forecasts/pseudo-measurements as PNG to `data/`.

## Project Structure

```
main.py                 # End-to-end example pipeline
tutorial.ipynb          # Interactive tutorial notebook
src/
├── download/           # Zenodo data download
├── load/               # Data loading, joining, resampling
├── model/              # Baseline models and evaluation metrics
│   ├── forecast.py
│   ├── pseudo_measurement_generation.py
│   └── _metrics.py
└── tests/              # Unit tests (pytest)
data/
└── zenodo/             # Downloaded dataset (gitignored)
```

## Testing

```bash
uv run pytest src/tests/
```

## Citation

If you use this dataset, preprint or code, please cite:

```bibtex
@misc{treutlein_real-world_2026,
	title = {Real-world energy data of 200 feeders from low-voltage grids with metadata in Germany over two years},
	url = {http://arxiv.org/abs/2602.03521},
	doi = {10.48550/arXiv.2602.03521},
	number = {{arXiv}:2602.03521},
	publisher = {{arXiv}},
	author = {Treutlein, Manuel and Bothe, Pascal and Schmidt, Marc and Hahn, Roman and Neumann, Oliver and Mikut, Ralf and Hagenmeyer, Veit},
	date = {2026-02-03},
	eprinttype = {arxiv},
	eprint = {2602.03521 [eess]},
	keywords = {Electrical Engineering and Systems Science - Systems and Control},
}
```

## License

This code is licensed under the [MIT License](LICENSE.txt).

Contact: Netze BW GmbH, Institute for Automation and Applied Informatics (IAI) (see corresponding e-mails in [preprint arXiv:2602.03521](https://arxiv.org/abs/2602.03521))