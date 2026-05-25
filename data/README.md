# Sample Datasets

| File | Records | Features | Task | Target | Source |
|------|---------|----------|------|--------|--------|
| `cs-training.csv` | 150,000 | 10 | Binary classification | `SeriousDlqin2yrs` | [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) |
| `UCI_Credit_Card.csv` | 30,000 | 23 | Binary classification | `default.payment.next.month` | [UCI ML Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) |

> **`loan_level_500k.csv` (Freddie Mac, 68 MB)** is excluded from this repo due to file size.  
> Download it directly from [Freddie Mac Single Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).

## Notes

- The GMSC dataset is sampled to 30,000 rows in the notebook for interactive speed.  
  The full 150 k rows are available in `cs-training.csv`.
- The Taiwan Credit dataset (`UCI_Credit_Card.csv`) drops the `ID` column automatically.
- All built-in sklearn datasets (Breast Cancer, Wine, Iris, Diabetes, California Housing)  
  are loaded directly from `sklearn.datasets` — no CSV required.
