"""
Jiajun Fang
DS 5010 Final Project
4/12/2024
"""
import pandas as pd
import numpy as np
# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
#  ski_gear_recommend import SkiGearRecommender
class happyski:
    """
    happyski is a package for ski lovers to train a ski resort rating system based on
	an existing dataset with ski resort conditions and its subjective ratings by other
	skiiers. After model training, user can ues the rating system to rate new ski resort
	data to provide them with a score from 1-5, and provide gear recommendation for users
	based on weather conditions.
    """
    def __init__(self, data_filepath, predict_filepath, model=None):
        """
        Initialize the ski resort dataest, prediction dataset, and model
        param data_filepath: file path for the main training dataset
        param predict_filepath: file path for the prediction dataset
        param model: the model we will later use to predict dataset (default is None)
        """
        print("Importing and preparing dataset...")
        self.dataset = pd.read_csv(data_filepath, encoding='cp1252') # Import main dataset
        self.dataset = self.clean_data(self.dataset) # Clean data
        self.prediction = pd.read_csv(predict_filepath, encoding='cp1252')  # Import prediction dataset
        self.model = model  # Initialize model
        print("Complete!")
    def clean_data(self, data):
        """
        Clean and transform dataset for model training.
        param data: a dataframe with all the variables
        returns: df_clean: a dataframe with only the necessary columns in it
        """
        # print("Cleaning and transforming data")
        df_encoded = pd.get_dummies(data, columns=['Skies', 'Snow Conditions']) # Get dummy variables
        df_clean = df_encoded.drop(['Date', 'Resort Name'], axis=1) # Dropping unimportant columns
        # print(df_clean.head()) # Check dataset format
        return df_clean # Return cleaned dataset
    def split_data(self):
        """
        Split data into X and y train and test.
        param: None. (Object method)
        returns: X_train, X_test, y_train, y_test for training and testing data
        """
        X = self.dataset.drop('Score', axis=1)  # Split X and y
        y = self.dataset['Score']

        # Splitting the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
        return X_train, X_test, y_train, y_test # Return training and testing dataset

    def train(self):
        """
        Train all the models and set the best performing model as object model.
        param: None. Object method
        returns: None
        """

        # Call in methods to train all the models
        print("\nTraining XGBoost Model")
        xbg_model, xbg_score = self.train_xgB()
        print("\nTraining Random Forest Model")
        rf_model, rf_score = self.train_rf()
        print("\nTraining Elastic Net Model")
        en_model, en_score = self.train_elastic_net()

        # Compare model score and select the model with the best performance as default model
        score_comparison = sorted([xbg_score, rf_score, en_score])
        if score_comparison[0] == xbg_score:
            self.model = xbg_model
            print("\nBest model is XGBoost")
        if score_comparison[0] == rf_score:
            self.model = rf_model
            print("\nBest model is Random Forest")
        if score_comparison[0] == en_score:
            self.mode = en_model
            print("\nBest model is Elastic Net")
    def train_rf(self):
        """
        Train random forest model using cleaned dataset.
        param: None. Object method.
        returns: classifier_rf, mse: the final model and its loss function score
        """
        # Split data for training
        X_train, X_test, y_train, y_test = self.split_data()

        # Initiate model
        regressor_rf = RandomForestRegressor(random_state=42)

        # Fitting model with Grid search CV
        params = {
            'n_estimators': [10 ,50 ,100],  # more trees may lead to better performance but consider computational cost
            'max_depth': [3, 5, None],  # allowing 'None' as an option to let some trees grow fully if needed
            'max_features': ['sqrt', 'log2', None],
        }
        # Negative mean squared error is commonly used in GridSearchCV for regression tasks
        scorer = make_scorer(mean_squared_error, greater_is_better=False)

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=regressor_rf,
                                   param_grid=params,
                                   cv=4,
                                   n_jobs=-1, verbose=1, scoring = scorer)

        # Find best model with grid search and save it
        grid_search.fit(X_train, y_train)
        best_rf_model = grid_search.best_estimator_

        # Testing model for performance
        y_pred = best_rf_model.predict(X_test)

        # Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)

        # Mean Absolute Error
        mae = mean_absolute_error(y_test, y_pred)
        print("Mean Absolute Error:", mae)

        # R-squared (Coefficient of Determination)
        r2 = r2_score(y_test, y_pred)
        print("R-squared:", r2)

        # Return model and score for comparison
        return best_rf_model, mse

    def train_xgB(self):
        """
        Train XGBoost model using cleaned dataset.
        param: None. Object method.
        returns: classifier_rf, mse: the final model and its loss function score
        """
        # Split data for training
        X_train, X_test, y_train, y_test = self.split_data()

        # Initiate model
        classifier_xgb = XGBRegressor(objective='reg:squarederror', n_estimators=50)

        # Fitting model with Grid search CV
        params = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }

        # Scorer for GridSearchCV using negative mean squared error
        scorer = "neg_mean_squared_error"

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=classifier_xgb,
                                   param_grid=params,
                                   cv=4,
                                   n_jobs=-1, verbose=1, scoring=scorer)

        # Find best model with grid search and save it
        grid_search.fit(X_train, y_train)
        best_xgb_model = grid_search.best_estimator_

        # Testing model for performance
        y_pred = best_xgb_model.predict(X_test)

        # Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)

        # Mean Absolute Error
        mae = mean_absolute_error(y_test, y_pred)
        print("Mean Absolute Error:", mae)

        # R-squared (Coefficient of Determination)
        r2 = r2_score(y_test, y_pred)
        print("R-squared:", r2)

        # Return model and score for comparison
        return best_xgb_model, mse

    def train_elastic_net(self):
        """
        Train linear regression model using cleaned dataset.
        param: None. Object method.
        returns: linear_model, mse: the final model and its loss function score
        """
        # Split data for training

        X_train, X_test, y_train, y_test = self.split_data()

        # Define the ElasticNet model
        elastic_net = ElasticNet()

        # Setup a parameter grid for GridSearchCV
        param_grid = {
            'alpha': [0.01, 0.1, 1, 10],  # Regularization strength; must be a positive float
            'l1_ratio': np.linspace(0.01, 1, 10),
            # Mixing parameter, with 0 being L2 penalty only and 1 being L1 penalty only
            'max_iter': [1000, 5000],  # Maximum number of iterations taken for the solvers to converge
            'tol': [0.0001, 0.001]  # Tolerance for the optimization
        }

        # Setup GridSearchCV
        grid_search_lr = GridSearchCV(estimator=elastic_net, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                                   verbose=1)

        # Fit GridSearchCV
        grid_search_lr.fit(X_train, y_train)

        # Best model
        best_model_lr = grid_search_lr.best_estimator_

        # Testing model for performance
        y_pred = best_model_lr.predict(X_test)

        # Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)

        # Mean Absolute Error
        mae = mean_absolute_error(y_test, y_pred)
        print("Mean Absolute Error:", mae)

        r2 = r2_score(y_test, y_pred)
        print("R-squared:", r2)

        # Return model and score for comparison
        return best_model_lr, mse
    def ski_model_rate(self):
        """
        Use object model to make prediction score based on user's new data.
        If user is unsatisfied with model rating, they can manually input a score of their own
        param: None. Object method.
        returns: None.
        """
        # Prepare prediction data for prediction
        prediction_clean = self.clean_data(self.prediction)

        # Drop score column
        X_prediction = prediction_clean.drop('Score', axis=1)

        # Add missing dummy variables (set = False)
        X_main_dataset = self.dataset.drop('Score', axis=1)  # Get trainset columns
        model_features = X_main_dataset.columns.tolist()
        missing_cols = set(model_features) - set(X_prediction.columns)
        for c in missing_cols:  # Add False to testing set columns
            X_prediction[c] = False
        # Ensure order of columns matches training dataset
        X_prediction = X_prediction[model_features]

        # Making prediction
        predicted_score = self.model.predict(X_prediction)

        # Print predicted score
        for i in range(len(predicted_score)):
            # Show ski resort data
            print(f"\nResort {i+1}'s Condition: ")
            print(self.prediction.drop('Score', axis=1).loc[i])
            print(f"\n{self.prediction.loc[i, 'Resort Name']}'s predicted rating is: {predicted_score[i]}\n")

            user_input = input("Are you satisfied with this score? (Y/N): ")
            if user_input == "Y":

                # Record prediction score to data
                self.prediction.loc[i, "Score"] = np.round(predicted_score[i], 2)
            else:
                # Ask user to input their score
                user_input = input(f"\nEnter your rating for {self.prediction.loc[i, 'Resort Name']}: ")
                self.prediction.loc[i, "Score"] = np.round(float(user_input), 2)

    def update(self, data_filepath):
        """
        Updates the new data and its prediction to the main dataset csv file
        param data_filepath: The file path that the predicted dataframe will be appended to
        returns: None
        """
        print("Updating predicted data to main dataset...")
        self.prediction.to_csv(data_filepath, mode='a', header=False, index=False)
        print("Complete!")






