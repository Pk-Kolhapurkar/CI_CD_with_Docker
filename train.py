import os
import pandas as pd
import skops.io as sio
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ensure required folders exist
os.makedirs("./Results", exist_ok=True)
os.makedirs("./Model", exist_ok=True)

# Load and shuffle the data
drug_df = pd.read_csv("Data/drug.csv")
drug_df = drug_df.sample(frac=1)

# Train-Test Split
X = drug_df.drop("Drug", axis=1).values
y = drug_df.Drug.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

# Define column types
cat_col = [1, 2, 3]
num_col = [0, 4]

# Build preprocessing pipeline
transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)
pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=10, random_state=125)),
    ]
)

# Train the model
pipe.fit(X_train, y_train)

# Model Evaluation
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")
print("Accuracy:", str(round(accuracy * 100, 2)) + "%", "F1:", round(f1, 2))

# Save metrics to a file
with open("./Results/metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}")

# Save the confusion matrix plot
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("./Results/model_results.png", dpi=120)

# Save the trained model
sio.dump(pipe, "./Model/drug_pipeline.skops")
#when you update file then do below steps in vscode terminal
#git commit -am "new changes"
#git push origin main
#trial
