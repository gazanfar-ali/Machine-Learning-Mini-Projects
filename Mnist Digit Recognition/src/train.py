from sklearn.model_selection import cross_val_score
from src.data_loader import load_mnist
from src.model import build_model, train_model
from src.visualization import plot_digit

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_mnist()

    # Binary classification: digit "2"
    y_train_2 = (y_train == 2)
    y_test_2 = (y_test == 2)

    # Build and train model
    model = build_model()
    model = train_model(model, X_train, y_train_2)

    # Evaluate
    scores = cross_val_score(model, X_train, y_train_2, cv=3, scoring="accuracy")
    print(f"Mean cross-validation accuracy: {scores.mean():.4f}")

    # Test visualization
    sample = X_train.iloc[36005]
    plot_digit(sample, label=y_train.iloc[36005])
    print("Prediction:", model.predict(sample.values.reshape(1, -1)))

if __name__ == "__main__":
    main()
