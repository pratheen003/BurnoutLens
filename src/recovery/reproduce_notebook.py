"""Faithful reproduction of original baseline notebook (data_understanding.ipynb).

Preserves all notebook steps and historical flaws:
1. Drop Person ID
2. Sleep Disorder fillna("None")
3. Blood Pressure split into Systolic and Diastolic while KEEPING Blood Pressure in X
4. Burnout Risk target derived from Stress Level
5. Lifestyle Score, Lifestyle Category, Burnout Index computed exactly as in notebook
6. X defined by dropping target and intermediate derived columns (keeps Blood Pressure)
7. LabelEncoder on every categorical/object column in X and target y
8. Data leakage flaw: StandardScaler fit_transform on ENTIRE X before train/test split
9. Stratified train_test_split (80/20, random_state=42)
10. Models evaluated: LogisticRegression, DecisionTree, RandomForest, XGBoost, MLPClassifier
11. 5-fold cross-validation evaluated on full scaled dataset without shuffling
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Suppress pandas string migration warning if applicable
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def reproduce_baseline():
    dataset_path = Path("data/raw/Sleep_health_and_lifestyle_dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    # Load with default pd.read_csv as notebook does
    df = pd.read_csv(dataset_path)

    # 1. Drop Person ID
    data = df.copy()
    data.drop("Person ID", axis=1, inplace=True)

    # 2. Fill missing sleep disorder values
    data["Sleep Disorder"] = data["Sleep Disorder"].fillna("None")

    # 3. Split Blood Pressure and KEEP original Blood Pressure column
    data[["Systolic BP", "Diastolic BP"]] = data["Blood Pressure"].str.split(
        "/", expand=True
    )
    data["Systolic BP"] = data["Systolic BP"].astype(int)
    data["Diastolic BP"] = data["Diastolic BP"].astype(int)

    # 4. Burnout Risk from Stress Level
    def classify_burnout(stress):
        if stress <= 4:
            return "Low"
        elif stress <= 6:
            return "Medium"
        else:
            return "High"

    data["Burnout Risk"] = data["Stress Level"].apply(classify_burnout)

    # 5. Functions copied exactly from notebook
    def calculate_lifestyle_score(row):
        score = 0
        if 7 <= row["Sleep Duration"] <= 9:
            score += 20
        elif 6 <= row["Sleep Duration"] < 7:
            score += 15
        else:
            score += 8

        score += (row["Quality of Sleep"] / 9) * 20
        score += (row["Physical Activity Level"] / 90) * 20
        score += min(row["Daily Steps"] / 10000, 1) * 20

        if 60 <= row["Heart Rate"] <= 75:
            score += 20
        elif 76 <= row["Heart Rate"] <= 80:
            score += 15
        else:
            score += 8

        return round(score, 2)

    data["Lifestyle Score"] = data.apply(calculate_lifestyle_score, axis=1)

    def lifestyle_category(score):
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Average"
        else:
            return "Poor"

    data["Lifestyle Category"] = data["Lifestyle Score"].apply(lifestyle_category)

    data["Burnout Index"] = (
        data["Stress Level"] * 10 + (100 - data["Lifestyle Score"])
    )

    # 6. Feature selection: drop derived intermediate/target columns, KEEP Blood Pressure
    X = data.drop(
        columns=[
            "Burnout Risk",
            "Stress Level",
            "Burnout Index",
            "Lifestyle Score",
            "Lifestyle Category",
        ]
    )
    y = data["Burnout Risk"]

    print("==================================================")
    print("STEP 5 - FAITHFUL NOTEBOOK REPRODUCTION")
    print("==================================================")
    print(f"Final X columns ({len(X.columns)} cols): {list(X.columns)}")
    print(f"Final X shape: {X.shape}")

    # 7. LabelEncoder on categorical columns and target
    label_encoders = {}
    categorical_columns = [
        col for col in X.columns if X[col].dtype == "object" or str(X[col].dtype).startswith("str")
    ]
    for column in categorical_columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    print(f"Target class mapping (index -> label): {dict(enumerate(target_encoder.classes_))}")
    print(f"Target classes: {list(target_encoder.classes_)}")

    # 8. StandardScaler fit_transform on ALL X before split (faithful reproduction of flaw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 9. Train/test split (80/20, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

    # 10. Model definitions exactly as used in notebook
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
        ),
        "MLP Classifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
            early_stopping=True,
        ),
    }

    results = {}
    reports = {}
    cv_results = {}
    lr_cm = None

    for name, model in models.items():
        print(f"\n--------------------------------------------------")
        print(f"Model: {name}")
        print(f"--------------------------------------------------")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        rep = classification_report(
            y_test, preds, target_names=target_encoder.classes_, digits=4
        )
        reports[name] = rep

        # 11. Cross-validation: 5-fold on full scaled array (notebook implementation)
        cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
        cv_results[name] = cv_scores

        print(f"Test Accuracy: {acc:.4f}")
        print(f"Classification Report:\n{rep}")
        print(f"CV Fold Scores: {np.round(cv_scores, 4)}")
        print(f"CV Mean Accuracy: {cv_scores.mean():.4f}")

        if name == "Logistic Regression":
            lr_cm = confusion_matrix(y_test, preds)
            print("\nLogistic Regression Confusion Matrix:")
            print(lr_cm)
            print(f"Labels order: {list(target_encoder.classes_)}")

    return {
        "X_columns": list(X.columns),
        "X_shape": X.shape,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "target_classes": list(target_encoder.classes_),
        "results": results,
        "cv_results": cv_results,
        "lr_confusion_matrix": lr_cm,
    }


if __name__ == "__main__":
    reproduce_baseline()
