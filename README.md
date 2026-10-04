# Neural Network in C

A feed-forward neural network built **from scratch in C**, with no ML libraries, as a way to understand how neural networks work at the lowest level. The current version classifies the classic **Iris** flower dataset.

## Features

- Network built from plain C structs (`Neuron`, `Layer`, `Network`)
- Forward propagation, backpropagation and weight updates implemented by hand
- Custom CSV reader to load the dataset (numeric and string columns)
- Label encoding of the species names into numeric targets
- Random shuffling of samples, with a train/test split (130 train / 20 test)
- Per-epoch error logging to a CSV file, for plotting the training curve
- Model saving to a text file after training
- Proper cleanup of allocated memory

## Project structure

```
.
├── main.c          # Training and testing pipeline
├── headers/
│   ├── activation.h  # Activation functions
│   ├── define.h      # Neuron / Layer / Network definitions
│   ├── init.h        # Layer and network initialization
│   ├── general.h     # Helpers (shuffle, error, max, save_model, ...)
│   └── csv_reader.h  # CSV parsing helpers
└── plot/
    ├── Iris.csv        # Dataset
    └── error_file.csv  # Training error per epoch (generated)
```

## Network architecture

| Layer  | Neurons | Inputs per neuron |
|--------|---------|-------------------|
| Input  | 4       | 4                 |
| Hidden | 10      | 4                 |
| Output | 3       | 10                |

The 4 inputs are the Iris measurements. The 3 output neurons correspond to the 3 species.

## Hyperparameters

Defined at the top of `main.c`:

| Name            | Value  |
|-----------------|--------|
| `learning_rate` | 0.001  |
| `EPOCHS`        | 274000 |

## Build and run

Requirements: a C compiler (GCC or Clang) and the math library.

```bash
git clone https://github.com/ycn3310/Neural-Network-in-C.git
cd Neural-Network-in-C
gcc main.c -o main -lm
./main
```

Run it from the repository root, since the program opens `plot/Iris.csv` and `plot/error_file.csv` using relative paths.

## What happens when you run it

1. Loads `plot/Iris.csv` and extracts the 4 feature columns and the species column
2. Encodes species as numeric targets: `Iris-setosa` → `0.0`, `Iris-versicolor` → `0.5`, `Iris-virginica` → `1.0`
3. Shuffles the samples, then holds out the last 20 for testing
4. Trains for `EPOCHS` epochs (forward pass, backpropagation, weight update for each sample)
5. Logs the average error of each epoch to `plot/error_file.csv` and prints progress every 10,000 epochs
6. Prints the output activations and error for each of the 20 test samples, plus the training time
7. Saves the trained model to `mark1.txt`

## Output files

- `plot/error_file.csv`: columns `epochs,error`, ready to plot
- `mark1.txt`: the saved model

## Roadmap

- [ ] Support more datasets
- [ ] Load a saved model and run predictions without retraining
- [ ] Add a plotting script for the error curve
- [ ] Make the architecture and dataset sizes configurable instead of hard-coded

## Motivation

This project is a learning exercise: implementing everything manually to deeply understand the concepts behind neural networks, instead of relying on a framework.
