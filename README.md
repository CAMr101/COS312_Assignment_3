# Multilayer Perceptron (MLP) Classifier

This project implements a Multilayer Perceptron (MLP) for binary classification using the Deeplearning4j (DL4J) library. It provides functionalities to train new models, perform hyperparameter grid search, and make predictions with existing trained models, all through a command-line interface.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Building the Project](#building-the-project)
- [Running the Application](#running-the-application)
- [Usage](#usage)
  - [1. Train a New Model](#1-train-a-new-model)
    - [1.1 Train a Single Model](#11-train-a-single-model)
    - [1.2 Run Full Grid Search](#12-run-full-grid-search)
  - [2. Make Predictions with Existing Model](#2-make-predictions-with-existing-model)
- [Data Format](#data-format)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Interactive Command-Line Interface**: Easy-to-navigate menu for different operations.
- **Single Model Training**: Train an MLP model with user-specified hyperparameters.
- **Hyperparameter Grid Search**: Automatically explore a range of hyperparameters.
- **Model Prediction**: Use saved models for predictions.
- **Data Loading & Preprocessing**: Load CSV data and apply standardization.
- **Model Persistence**: Save trained models for future use.

## Technologies Used

- Java 17+
- Deeplearning4j (DL4J)
- ND4J
- Maven

## Prerequisites

- Java Development Kit (JDK) 17 or higher
- Apache Maven 3.6.0 or higher

## Project Structure

```
.
├── pom.xml
└── src
    └── main
        └── java
            └── com
                └── assignment
                    └── mlp
                        ├── App.java
                        ├── DataLoader.java
                        ├── GridSearchExecutor.java
                        ├── HyperparameterConfig.java
                        ├── MultilayerPerceptron.java
                        ├── UserInput.java
                        └── UtilityFunctions.java
```

## Getting Started

### Building the Project

Clone the repository:

```bash
git clone https://github.com/CAMr101/COS312_Assignment_3
cd COS312_Assignment_3
```

Build using Maven:

```bash
mvn clean install
```

### Running the Application

```bash
java -jar target/mlp-classifier-1.0-SNAPSHOT.jar
```

## Usage

Upon running the application, you'll see:

```
=== MLP Application ===
1. Train a new model
2. Make predictions with existing model
3. Exit
Enter your choice (1, 2 or 3):
```

### 1. Train a New Model

#### 1.1 Train a Single Model

Parameters:

- Path to the CSV file
- Random seed (default: 527)
- Learning rate (default: 0.0005)
- Batch size (default: 128)
- Epochs (default: 150)
- Hidden layer neurons (default: 64 each)
- Activation function (default: TANH)
- Weight initialization (default: XAVIER)

#### 1.2 Run Full Grid Search

Prompts:

- Path to training CSV
- Base random seed

**Note**: Resource-intensive and time-consuming.

### 2. Make Predictions with Existing Model

Prompts:

- Path to saved model file
- Path to prediction CSV

Outputs predictions with class and probability.

## Data Format

- First line: header (skipped)
- Comma-separated values
- 6 columns:
  - First 5: features
  - 6th: binary label (0 or 1)

## Contributing

Contributions are welcome! Fork the repo, open issues, or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
