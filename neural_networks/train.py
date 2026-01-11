import numpy as np
from neural_network import NeuralNetwork, one_hot_encode, normalize
from tqdm import tqdm


def create_data(samples=1000, features=2, num_labels=2):
    np.random.seed(42)

    # Raw input
    X = np.random.randn(features, samples)

    # Normalize using YOUR function
    X = normalize(X)

    # Create labels
    y_raw = (X[0] + X[1] > 1).astype(int)

    # One-hot encode using YOUR function
    y = one_hot_encode(y_raw, num_labels).T   # (classes, samples)

    # Train-test split
    split = int(0.8 * samples)

    X_train = X[:, :split]
    y_train = y[:, :split]

    X_test = X[:, split:]
    y_test = y[:, split:]

    return X_train, y_train, X_test, y_test


def main():
    # -----------------------------
    # Dataset
    # -----------------------------
    X_train, y_train, X_test, y_test = create_data()

    # -----------------------------
    # Network architecture
    # -----------------------------
    architecture = [16, 8]   # hidden layers only

    # -----------------------------
    # Model
    # -----------------------------
    model = NeuralNetwork(
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test,
        activation="relu",
        num_labels=2,
        architecture=architecture
    )

    # -----------------------------
    # Training
    # -----------------------------
    model.fit(
        X_train,
        y_train,
        lr=0.01,
        epochs=500
    )

    # -----------------------------
    # Evaluation
    # -----------------------------
    print("\nFinal Results")
    print("-------------------------")
    print(f"Train Accuracy: {model.accuracy(X_train, y_train):.2f}%")
    print(f"Test Accuracy : {model.accuracy(X_test, y_test):.2f}%")


if __name__ == "__main__":
    main()

