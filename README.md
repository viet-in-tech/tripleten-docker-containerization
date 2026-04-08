# Titanic Survival Prediction API

A machine learning API that predicts passenger survival on the Titanic using a Random Forest classifier.

## Quick Start

### Run the API locally

docker pull yourusername/titanic-survival-api:latest
docker run -p 8000:8000 yourusername/titanic-survival-api:latest


### Use the API

Visit http://localhost:8000/docs for interactive documentation.

#### Example prediction:

curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Pclass": 1,
       "Sex": "female", 
       "Age": 30.0,
       "SibSp": 0,
       "Parch": 0,
       "Fare": 50.0,
       "Embarked": "S"
     }'


#### Response:

{
  "survived": 1,
  "survival_probability": 0.95,
  "prediction_confidence": "High",
  "passenger_profile": {
    "class": "Class 1",
    "gender": "female",
    "age_group": "Adult",
    "family_size": 1,
    "traveling_alone": true,
    "fare_level": "High",
    "embarkation_port": "Southampton"
  }
}


## Model Performance

- Algorithm: Random Forest Classifier
- Accuracy: 85%+ on test data
- Features: Passenger class, sex, age, family size, fare, embarkation port

## API Endpoints

-  GET /               - Health check
-  POST /predict       - Single passenger prediction
-  POST /predict-batch - Multiple passenger predictions
-  GET /model-info     - Model metadata

## Input Parameters

| Parameter | Type  |       Description            |  Example |
|-----------|-------|------------------------------|----------|
| Pclass    |  int  | Passenger class (1, 2, 3)    | 1        |
| Sex       |  str  | Gender ('male', 'female')    | 'female' |
| Age       | float | Age in years                 | 30.0     |
| SibSp     |  int  | Siblings/spouses aboard      | 0        |
| Parch     |  int  | Parents/children aboard      | 0        |
| Fare      | float | Ticket fare                  | 50.0     |
| Embarked  |  str  | Port ('S', 'C', 'Q')         | 'S'      |

## Development

### Training Environment

cd training
docker build -t titanic-training .
docker run -p 8888:8888 -v "$(pwd):/home/jovyan/work" titanic-training


### Local Development

cd serving
pip install -r requirements.txt
python app.py


## License

This project is for educational purposes.
