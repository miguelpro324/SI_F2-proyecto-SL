# SI_F2-proyecto-SL

[creditcard.csv](./download-creditcard-csv.sh) recuperado de [Kaggle.com](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Ejecutar la UI

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Ejecutar scripts standalone

```bash
python logistic_regression.py
python svm.py
python decision_tree.py
```
