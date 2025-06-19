# Iterative training of an XGBoost model

## Run
```sh
pip install -r requirements.txt
python predict_features_cut.py
```

## Using other data
Another dataset can be used to train the model by editing the `predict_features_cut.py` file. Change the path specified at line 216.

The species in the dataset must have any name other than 'REFERENCE', 'SAMPLE CODE', 'SITE NAME', 'pH', 'LAT', 'LONG', 'WTD', 'PERSON', 'COUNTRY', 'SITE', 'SAMPLE' or 'Unnamed'. Any value named otherwise will be selected as features for the new model.

The column for the target value must be named 'WTD'.

The code should work normally with any other dataset.

## Dataset filtering
A filter is applied to the dataset in the `process.py` file on line 11. The current filter keeps any line where 'WTD' is between -10 and 70 and 'pH' is less or equal to 5.5. This piece of code can be edited to apply another filter.