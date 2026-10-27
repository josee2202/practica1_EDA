# Data

The data files are not versioned. Download them and place them as shown below.

## Source

[Credit EDA Case Study](https://www.kaggle.com/datasets/venkatasubramanian/credit-eda-case-study) on Kaggle
(listed there under CC0; the records derive from the Home Credit Default Risk data). Only two files are used:

| File | Size | Used by |
|---|---|---|
| `application_data.csv` | 166,133,370 bytes, 307,511 rows x 122 columns | all notebooks |
| `columns_description.xlsx` | ~19 KB | reference only (column dictionary) |

SHA-256 of the `application_data.csv` used for the published results:
`52e96b895b1112e1c853f670e58372719c8441c5ed1c57ac2f7fad559d784f5f`

```bash
# with the Kaggle CLI (needs a Kaggle API token in ~/.kaggle/kaggle.json)
kaggle datasets download venkatasubramanian/credit-eda-case-study -f application_data.csv -p data/raw_data
# if it arrives zipped: unzip data/raw_data/application_data.csv.zip -d data/raw_data
shasum -a 256 data/raw_data/application_data.csv   # or: sha256sum on Linux
```

Or download `application_data.csv` from the dataset page and copy it to `data/raw_data/`.

## Layout

```
data/
├── raw_data/application_data.csv        # input (download)
├── processed_data/                      # written by notebooks 02 and 03
│   ├── train_data_processed.csv         # imputed train split (80%)
│   ├── test_data_processed.csv          # imputed test split (20%)
│   ├── df_train_scaled.csv              # encoded, scaled, selected features + TARGET
│   └── df_test_scaled.csv
└── variables/                           # small lists and encodings written by notebooks 01, 03, 04
```

For CI and quick checks, `scripts/make_synthetic_data.py` writes a synthetic file with the same schema
(random values, hand-written target). It is only for testing that the code runs.
