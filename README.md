# Multilayer Perceptron (MLP) Classifier

This project implements a Multilayer Perceptron (MLP) for binary classification using the Deeplearning4j (DL4J) library. It provides functionalities to train new models, perform hyperparameter grid search, and make predictions with existing trained models, all through a command-line interface.

---

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

---

## Features

- **Interactive Command-Line Interface**: Easy to navigate menu for different operations.
- **Single Model Training**: Train an MLP model with user-specified hyperparameters (data path, random seed, batch size, learning rate).
- **Hyperparameter Grid Search**: Automatically explore a predefined range of hyperparameters to find the best performing model. Results are saved to a CSV file (`grid_search_results.csv`).
- **Model Prediction**: Load a previously saved model and use it to make predictions on new datasets.
- **Data Loading & Preprocessing**: Handles loading CSV data and applies standardization for optimal model performance.
- **Model Persistence**: Option to save trained models to a file for later use.

---

## Technologies Used

- **Java 17+**
- **Deeplearning4j (DL4J)** - Deep learning library for Java.
- **ND4J** - Numerical computing library for Java, used by DL4J.
- **Maven** - Project management and dependency handling.

---

## Prerequisites

Before you begin, ensure you have the following installed:

- Java Development Kit (JDK) 17 or higher
- Apache Maven 3.6.0 or higher

---

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
                        └── UserInput.java
```

- **App.java**: Main application entry point.
- **DataLoader.java**: Loads CSV datasets, normalizes data, collects user input.
- **GridSearchExecutor.java**: Manages hyperparameter grid search and saves results.
- **HyperparameterConfig.java**: POJO for model hyperparameters.
- **MultilayerPerceptron.java**: Builds the DL4J `MultiLayerNetwork`.
- **UserInput.java**: Stores user-provided input parameters.

---

## Getting Started

### Building the Project

Clone the repository:

```bash
git clone <repository_url>
cd mlp-classifier
```

(Replace `<repository_url>` with the actual URL of your repository)

Build using Maven:

```bash
mvn clean install
```

This compiles the source code, runs tests, and packages the app into a JAR file in the `target/` directory.

### Running the Application

After building, run the application:

```bash
java -jar target/mlp-classifier-1.0-SNAPSHOT.jar
```

(Adjust the JAR name if different.)

---

## Usage

Upon running the application, you will see:

```
Choose mode:
1. Train a new model
2. Make predictions with existing model
3. Exit
Enter your choice (1, 2 or 3):
```

### 1. Train a New Model

Choose:

```
Choose Training mode:
1. Train a single model
2. Run full grid search
3. Exit
Enter your choice (1, 2 or 3):
```

#### 1.1 Train a Single Model

Prompts:

- Path to the CSV file
- Random seed (default: 527)
- Batch size (default: 32)
- Learning rate (default: 0.001)

Example:

```
Enter the path to the CSV file (relative or absolute): data/train.csv
Enter a random seed (default 527):
Enter batch size (default 32):
Enter learning rate (default 0.001):
```

After training, stats will display, and you can save the model.

#### 1.2 Run Full Grid Search

Prompts:

- Path to training data CSV
- Optional random seed

Example:

```
Enter path to training data CSV file: data/train.csv
Enter base random seed (default 527):
```

> ⚠️ Grid search is resource-intensive and may take time.

Results saved in `grid_search_results.csv`.

### 2. Make Predictions with Existing Model

Prompts:

- Path to saved model file
- Path to CSV for predictions

Example:

```
Enter path to saved model file: myModel.zip
Enter path to CSV file for predictions: data/predict.csv
```

Displays predicted class and probability for each sample.

---

## Data Format

The application expects CSV files:

- First line (header) is skipped
- Comma (`,`) delimiter
- 6 columns:
  - 1st to 5th columns (indices 0-4) are features
  - 6th column (index 5) contains binary labels

Ensure your CSV adheres to this format.

---

## Contributing

Contributions are welcome! Fork the repo, open issues, or submit pull requests.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
