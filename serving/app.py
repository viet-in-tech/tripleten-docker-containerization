from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict passenger survival on the Titanic using machine learning",
    version="1.0.0"
)

# Load model and preprocessing artifacts
MODEL_PATH = "models/titanic_model.joblib"
ENCODERS_PATH = "models/label_encoders.joblib"
METADATA_PATH = "models/model_metadata.json"

try:
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    print("Model and preprocessing artifacts loaded successfully!")
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    model = None
    label_encoders = None
    metadata = None

class PassengerInput(BaseModel):
    Pclass: int  # 1, 2, or 3
    Sex: str  # 'male' or 'female'
    Age: float  # Age in years
    SibSp: int  # Number of siblings/spouses aboard
    Parch: int  # Number of parents/children aboard
    Fare: float  # Fare paid
    Embarked: str  # 'S', 'C', or 'Q'

    class Config:
        json_schema_extra = {
            "example": {
                "Pclass": 3,
                "Sex": "male",
                "Age": 25.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 8.05,
                "Embarked": "S"
            }
        }

class PredictionOutput(BaseModel):
    survived: int  # 0 or 1
    survival_probability: float  # Probability of survival
    prediction_confidence: str  # High, Medium, or Low
    passenger_profile: Dict[str, Any]  # Passenger characteristics

def preprocess_passenger(passenger_data: dict) -> pd.DataFrame:
    """Preprocess passenger data to match training format"""

    # Create DataFrame from input
    df = pd.DataFrame([passenger_data])

    # Create derived features (same as training)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Create Title feature (simplified for prediction)
    sex = passenger_data['Sex']
    if sex == 'male':
        df['Title'] = 'Mr'
    else:
        age = passenger_data['Age']
        df['Title'] = 'Mrs' if age >= 18 else 'Miss'

    # Create age groups
    age = passenger_data['Age']
    if age <= 12:
        age_group = 'Child'
    elif age <= 18:
        age_group = 'Teen'
    elif age <= 35:
        age_group = 'Adult'
    elif age <= 60:
        age_group = 'Middle'
    else:
        age_group = 'Senior'
    df['AgeGroup'] = age_group

    # Create fare groups (simplified)
    fare = passenger_data['Fare']
    if fare <= 7.91:
        fare_group = 'Low'
    elif fare <= 14.45:
        fare_group = 'Medium'
    elif fare <= 31.0:
        fare_group = 'High'
    else:
        fare_group = 'VeryHigh'
    df['FareGroup'] = fare_group

    # Encode categorical variables using saved encoders
    categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']

    for col in categorical_cols:
        if col in label_encoders:
            try:
                # Handle unseen categories by using the most frequent class
                if df[col].iloc[0] in label_encoders[col].classes_:
                    df[col] = label_encoders[col].transform(df[col])
                else:
                    # Use the most frequent class for unseen categories
                    df[col] = 0  # Assuming 0 is typically the most frequent
            except Exception as e:
                print(f"Error encoding {col}: {e}")
                df[col] = 0

    # Ensure all required features are present in correct order
    required_features = metadata['feature_names']
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0  # Default value for missing features

    return df[required_features]

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Titanic Survival Prediction API",
        "status": "healthy" if model is not None else "error",
        "model_loaded": model is not None,
        "accuracy": metadata['accuracy'] if metadata else None
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_survival(passenger: PassengerInput):
    """Predict passenger survival probability"""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Preprocess input data
        passenger_df = preprocess_passenger(passenger.dict())

        # Make prediction
        survival_prob = model.predict_proba(passenger_df)[0]
        survived = int(model.predict(passenger_df)[0])
        survival_probability = float(survival_prob[1])  # Probability of survival

        # Determine confidence level
        if survival_probability > 0.8 or survival_probability < 0.2:
            confidence = "High"
        elif survival_probability > 0.6 or survival_probability < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Create passenger profile
        profile = {
            "class": f"Class {passenger.Pclass}",
            "gender": passenger.Sex,
            "age_group": "Child" if passenger.Age < 18 else "Adult",
            "family_size": passenger.SibSp + passenger.Parch + 1,
            "traveling_alone": passenger.SibSp + passenger.Parch == 0,
            "fare_level": "High" if passenger.Fare > 30 else "Medium" if passenger.Fare > 15 else "Low",
            "embarkation_port": {
                'S': 'Southampton',
                'C': 'Cherbourg',
                'Q': 'Queenstown'
            }.get(passenger.Embarked, 'Unknown')
        }

        return PredictionOutput(
            survived=survived,
            survival_probability=survival_probability,
            prediction_confidence=confidence,
            passenger_profile=profile
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")

    return {
        "model_type": metadata['model_type'],
        "training_date": metadata['training_date'],
        "accuracy": metadata['accuracy'],
        "features": metadata['feature_names'],
        "preprocessing": metadata['preprocessing_info']
    }

@app.post("/predict-batch")
async def predict_batch(passengers: list[PassengerInput]):
    """Predict survival for multiple passengers"""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = []
        for passenger in passengers:
            # Use the single prediction endpoint for each passenger
            prediction = await predict_survival(passenger)
            results.append({
                "passenger": passenger.dict(),
                "prediction": prediction.dict()
            })

        return {"predictions": results, "count": len(results)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
