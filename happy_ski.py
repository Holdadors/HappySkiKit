"""
Jiajun Fang
DS 5010 Final Project
4/12/2024

"""
import pandas as pd
import numpy as np
# ML libraries
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS # Can replace using gridsearchCV

class happyski:
    """
    Package that allows user to:
    1. train a machine learning ski resort rating system
    2. Rating new ski resort data
    3. Update models based on new predicted data
    4. Allow users to make custom ratings if they are dissatisfied wth the predicted ratings
    """
    def __init__(self, data_filepath, predict_filepath, model=None):
        """Have default models and dataset ready"""
        self.dataset = pd.read_csv(data_filepath, encoding='cp1252')  # Saves the main excel file
        # Clean data
        self.dataset = self.clean_data(self.dataset)

        self.prediction = pd.read_csv(predict_filepath, encoding='cp1252')  # Saves user dataframe/datapoint
        # Clean prediction
        # self.prediction = self.clean_data(self.prediction)

        self.model = model  # Saves the ML model
    def clean_data(self, data):
        """Import and clean data for modelling"""
        print("Cleaning and transforming data")
        df_encoded = pd.get_dummies(data, columns=['Skies', 'Snow Conditions'])  # Get dummy variables
        df_clean = df_encoded.drop(['Date', 'Resort Name'], axis=1)  # Dropping datetime and resort name
        print(df_clean.head())  # Check dataset format
        return df_clean
    def split_data(self):
        """Split data into X and y train and test"""
        # Putting feature variable to X
        # clean_data = self.clean_data(self.dataset)
        # self.dataset = clean_data
        X = self.dataset.drop('Score', axis=1)
        # Putting response variable to y
        y = self.dataset['Score']
        # Splitting the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
        return X_train, X_test, y_train, y_test

    def train(self):
        """Train best model to predict rating"""
        # Call in methods to train all the models
        rf_model, rf_score = self.train_rf()
        lr_model, lr_score = self.train_lr()

        # Compare model score
        # Select model with the best performance as default model
        if rf_score > rf_score:
            self.model = rf_model
        else:
            self.model = lr_model
        print(self.model)

    def train_rf(self):
        """Train Random forest model"""
        X_train, X_test, y_train, y_test = self.split_data()
        classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                               n_estimators=50, oob_score=True)

        # Fitting model (Replace cv grid search here)
        classifier_rf.fit(X_train, y_train)
        # Predicting model
        y_pred = classifier_rf.predict(X_test)

        # Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)

        # Mean Absolute Error
        mae = mean_absolute_error(y_test, y_pred)
        print("Mean Absolute Error:", mae)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)

        # Classification Report (Precision, Recall, F1-Score)
        class_report = classification_report(y_test, y_pred)
        print("Classification Report:\n", class_report)

        # Return model and score for comparison
        return classifier_rf, mse
    def train_lr(self):
        """Train Linear Regression model"""
        X_train, X_test, y_train, y_test = self.split_data()
        linear_model = LinearRegression().fit(X_train, y_train) # Replace with cv grid search

        # Predicting model
        y_pred = linear_model.predict(X_test)

        # Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)

        # Mean Absolute Error
        mae = mean_absolute_error(y_test, y_pred)
        print("Mean Absolute Error:", mae)

        # Return model and score for comparison
        return linear_model, mse
    def ski_model_rate(self):
        """Make a prediction based on input data"""
        # Input x variable (single row or dataframe), transform it for model use
        # After making predictions with current model, ask if user is satisfied
        # If not satisfied, call in user_predict for them to override
        # clean_data = self.clean_data(self.prediction)
        # Clean prediction data
        # self.prediction = clean_data
        prediction_clean = self.clean_data(self.prediction)
        # Make prediction with existing model
        X = prediction_clean.drop('Score', axis=1)
        # Add missing dummy variables (set = False)
        X_train = self.dataset.drop('Score', axis=1)  # Get trainset columns
        model_features = X_train.columns.tolist()
        missing_cols = set(model_features) - set(X.columns)
        for c in missing_cols:  # Add False to testing set columns
            X[c] = False
        # Ensure order of columns matches training dataset
        X = X[model_features]
        print(X)
        # Making prediction
        predicted_score = self.model.predict(X)
        # Print predicted score
        print(f"Predicted: {predicted_score}")

        user_input = input("Are you satisfied with this score? (Y/N): ")
        if user_input == "Y":

            # Record prediction score to data
            self.prediction["Score"] = np.round(predicted_score, 2)
        else:
            self.user_rate()
    def user_rate(self):
        """If user not satisfied, can manually override the prediction score"""
        # Use loop to show x variable, and allow user to input Y variable
        user_input = input("Enter your rating: ")
        self.prediction["Score"] = user_input

    def update(self, data_filepath):
        """Allow user to update data by giving their own scores"""
        # Update prediction score onto main excel cell by writing it in as newest datapoint
        # Write score onto prediction csv, then combine this csv to the main csv
        # Problem: prediction dataset does not have all the dummy variables cause it only has one entry
        # Solution: Make prediction have the same columns as dataset (if no values, input false)
        self.prediction.to_csv(data_filepath, mode='a', header=False, index=False)

if __name__ == "__main__":
    # Creating an object
    user_happyski = happyski("snow.csv", "snow prediction.csv")

    # Check clean data
    # clean_data = user_happyski.clean_data(user_happyski.dataset)
    # clean_test = user_happyski.clean_data(user_happyski.prediction)

    # Check Split
    # collection_x_y = user_happyski.split_data()
    # print(collection_x_y)

    # Check train rf
    # rf = user_happyski.train_rf()
    # print(rf)
    # Check train lr
    # lr = user_happyski.train_lr()
    # print(lr)

    # Check train model comparison
    user_happyski.train()

    # Check prediction
    user_happyski.ski_model_rate()
    print(user_happyski.prediction["Score"])
    # Check update
    user_happyski.update("snow.csv")





