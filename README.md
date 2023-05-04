# AlgoEconFInal

## Generating Training Data

Run `python3 gen_script.py` to generate the test data into test_data.npy. 

## Loading Training Data 
```
with open("test_data.npy", 'rb') as f:
    data = np.load(f)
```