import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def main():

    print("Creating models folder...")
    os.makedirs("models", exist_ok=True)

    print("Generating dummy dataset...")
    X = np.random.rand(200, 5)
    y = np.random.randint(0, 2, 200)

    print("Training model...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_scaled, y)

    print("Saving model files...")
    with open("models/random_forest.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Model saved successfully!")

if __name__ == "__main__":
    main()