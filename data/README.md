# Real transaction data

Source: [ULB / Worldline Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
The original file contains 284,807 transactions, including 492 frauds, from European cardholders
over two days in September 2013. `Class=1` means fraud; `Class=0` means legitimate.
`Time` is seconds since the first recorded transaction, `Amount` is transaction amount,
and `V1`–`V28` are anonymized PCA components. The original business features and PCA transform
are unavailable, so ordinary card details cannot be converted into these components here.

After installing dependencies, run:

```powershell
python scripts/download_data.py
```

This uses the public CSV mirror linked in [TensorFlow's official tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).
It checks the schema, record count, and fraud count, then records a SHA256 in `creditcard.source.json`.

If automatic downloading is unavailable:

1. Open the Kaggle source link above and sign in if requested.
2. Download the dataset ZIP and extract `creditcard.csv`.
3. Place it at `data/creditcard.csv` (the file must have `Time,V1,...,V28,Amount,Class`).
4. Run `python scripts/train.py`.

The data and binary model files are ignored by Git. Observe the dataset's license on Kaggle
when sharing data. No fabricated transactions are used for reported experiments.
