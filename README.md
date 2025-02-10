# project_m3_MachineLearning
### Diamond Price Prediction | Supervised Machine Learning  
_Kaggle competition to get minimum RMSE_

#### Overview   
This project aims to predict the price of diamonds using various features like carat weight, cut, clarity, color, and others. The goal is to build a regression model that can accurately predict the price of diamonds based on their characteristics.

The training dataset contains information about over 40,455 diamonds and their respective features. Various machine learning models, with different combinations of pre-processing techniques, were trained and evaluated to optimize the prediction accuracy. Then the models were used to make predictions on different dataset for Kaggle competition, evaluating the universality of the trained model.

#### Dataset  
The dataset used in this project is the Diamonds dataset, which includes the following columns:

* carat: Weight of the diamond (in carats)  
* cut: Quality of the cut (Ordinal: Fair, Good, Very Good, Premium, Ideal)  
* clarity: Clarity of the diamond (Ordinal: I1, SI1, SI2, VS1, VS2, VVS1, VVS2, IF)  
* color: Color of the diamond (Ordinal: J, I, H, G, F, E, D)  
* city: City where the diamond was sold  
* depth: Depth percentage of the diamond  
* table: Table percentage of the diamond  
* x, y, z: Dimensions of the diamond (length, width, height)  
* price: Price of the diamond (target variable)  
The dataset includes categorical variables such as cut, clarity, and color that require preprocessing (e.g., one-hot encoding or label encoding).

Additional Information:  
Training Dataset: Contains 40,455 diamonds with both features and prices. The Exploratory Data Analysis was performed upfront and can be found [here.](https://github.com/Kristinawk/EDA_Diamonds/blob/main/notebooks/EDA.ipynb)  
Test Dataset: Contains 13,485 records for predicion (price is not given).

#### Project Structure
```md
project_m3_MachineLearning/
├── data/
│   ├── diamonds_train.csv              # Training dataset
│   ├── diamonds_test.csv               # Test dataset (no target variable)
│
├── notebooks/
│   ├── functions.py                    # Helper functions for data preprocessing
│   ├── encod03_featu00_model02.ipynb   # Model training - best Kaggle score 
│   ├── ...                             # Model training - other attempts 
│
├── predictions/
│   ├── encod03_featu00_model02.csv     # Model prediction for test dataset 
│
├── .gitignore
└── README.md                           # Project documentation (this file)
```

#### Requirements
This project requires Python and several libraries: 

```pandas```   
```numpy```  
```scikit-learn```   
```xgboost```  
```matplotlib```  
```seaborn```

#### Data Preprocessing
Steps:  
1. Create functions.py: Automate the most repetitive processing and training tasks.   
2. Load the dataset: Load the training and test datasets into pandas DataFrames.  
3. Outliers management: Analyse outliers in numerical features and clean dataset.  
4. Feature encoding: Encode cut, color, and clarity using LabelEncoder. Drop city.  
Refer to ```notebooks/encod03_featu00_model02.ipynb``` for the full data preprocessing workflow.

#### Model Training

Models Used:  

Random Forest Regressor  
Extra Trees Regressor  
XGBoost Regressor  
Stacking Regressor (combining multiple models)  

Steps:  

1. Train-test data sets: Split preprocessed training dataset into train and test (20%) subsets.  
2. Evaluate multiple models: Perform cross-validation with various models. Refer to ```notebooks/encod01_featu00_model01.ipynb``` and ```notebooks/encod03_featu00_model02.ipynb```.  
3. Stacking Regressor: Combine base models (Random Forest, XGBoost, Extra Trees) using a meta-model (Ridge) to improve prediction accuracy.  
4. Train the model: Train Stacking Regressor on the preprocessed training dataset.

#### Model Prediction  
Use trained model to predict diamonds prices from the test dataset. The most accurate prediction was saved in  ```predictions/encod03_featu00_model02.ipynb``` and uploaded into Kaggle. The RMSE achieved in training dataset was 525$, and 558$ in test dataset (train dataset price mean = 3,928$, IQR = 4,386$)

#### Saving and Loading the Model
After training, you can save the trained model using joblib for later use.

```
import joblib

# Save model
joblib.dump(model, 'trained_model.pkl')  
```
To load the saved model for predictions:
```
model = joblib.load('trained_model.pkl')
