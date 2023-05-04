# AlgoEconFInal

## Generating Training Data

Run `python3 gen_script.py <num_graphs> <seed>` to generate the test data into test_data.npy. 

## Loading Training Data 
```
with open("test_data_COUNT_SEED.npy", 'rb') as f:
    data = np.load(f)
```

## Training the Model
Run `python3 simulation.py train <test_filename.npy>` to train the model.