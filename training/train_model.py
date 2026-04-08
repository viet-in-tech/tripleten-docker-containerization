import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data():
    """Load and explore the Titanic dataset"""
    print("Loading Titanic dataset...")
    df = pd.read_csv('data/titanic.csv')

    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSurvival rate by gender:")
    print(df.groupby('Sex')['Survived'].mean())

    print("\nSurvival rate by class:")
    print(df.groupby('Pclass')['Survived'].mean())

    return df

def preprocess_data(df):
    """Clean and preprocess the data"""
    print("\nPreprocessing data...")

    # Create a copy for processing
    data = df.copy()

    # Handle missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)

    # Create new features
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    # Extract title from name
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                          'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                          'Jonkheer', 'Dona'], 'Other')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    # Age groups
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100],
                             labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

    # Fare groups
    data['FareGroup'] = pd.qcut(data['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])

    print(f"Preprocessed data shape: {data.shape}")
    print("\nNew features created:")
    print("- FamilySize: SibSp + Parch + 1")
    print("- IsAlone: Whether passenger traveled alone")
    print("- Title: Title extracted from name")
    print("- AgeGroup: Age categories")
    print("- FareGroup: Fare categories")

    return data

def encode_features(data):
    """Encode categorical features"""
    print("\nEncoding categorical features...")

    # Select features for modeling
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup']

    model_data = data[features + ['Survived']].copy()

    # Initialize label encoders
    label_encoders = {}

    # Encode categorical variables
    categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']

    for col in categorical_cols:
        le = LabelEncoder()
        model_data[col] = le.fit_transform(model_data[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {list(le.classes_)}")

    return model_data, label_encoders

def train_model(model_data):
    """Train the Random Forest model"""
    print("\nTraining Random Forest model...")

    # Prepare features and target
    X = model_data.drop('Survived', axis=1)
    y = model_data['Survived']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model, accuracy, feature_importance, X.columns.tolist()

def save_model_artifacts(model, label_encoders, feature_names, accuracy):
    """Save model and preprocessing artifacts"""
    print("\nSaving model artifacts...")

    # Save the trained model
    model_filename = 'models/titanic_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved to: {model_filename}")

    # Save label encoders
    encoders_filename = 'models/label_encoders.joblib'
    joblib.dump(label_encoders, encoders_filename)
    print(f"Label encoders saved to: {encoders_filename}")

    # Save model metadata
    metadata = {
        'model_type': 'RandomForestClassifier',
        'training_date': datetime.now().isoformat(),
        'accuracy': float(accuracy),
        'feature_names': feature_names,
        'model_parameters': model.get_params(),
        'preprocessing_info': {
            'age_fillna': 'median',
            'embarked_fillna': 'mode',
            'fare_fillna': 'median',
            'new_features': ['FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup']
        }
    }

    metadata_filename = 'models/model_metadata.json'
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_filename}")

    # Create a sample prediction for testing
    sample_passenger = {
        'Pclass': 3,
        'Sex': 'male',
        'Age': 25.0,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 8.05,
        'Embarked': 'S',
        'FamilySize': 1,
        'IsAlone': 1,
        'Title': 'Mr',
        'AgeGroup': 'Adult',
        'FareGroup': 'Low'
    }

    sample_filename = 'models/sample_input.json'
    with open(sample_filename, 'w') as f:
        json.dump(sample_passenger, f, indent=2)
    print(f"Sample input saved to: {sample_filename}")

    print("\nAll model artifacts saved successfully!")
    return metadata

def main():
    """Main training pipeline"""
    print("Starting Titanic Survival Prediction Model Training")
    print("=" * 60)

    # Load and explore data
    df = load_and_explore_data()

    # Preprocess data
    processed_data = preprocess_data(df)

    # Encode features
    model_data, label_encoders = encode_features(processed_data)

    # Train model
    model, accuracy, feature_importance, feature_names = train_model(model_data)

    # Save artifacts
    metadata = save_model_artifacts(model, label_encoders, feature_names, accuracy)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Final model accuracy: {accuracy:.4f}")
    print("Model artifacts saved to 'models/' directory")
    print("Ready for deployment!")

if __name__ == "__main__":
    main()
