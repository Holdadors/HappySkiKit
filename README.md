# HappySkiKit
## Overview
HappySkiKit is a machine-learning framework designed to optimize the ski resort experience by offering dynamic slope ratings and personalized gear recommendations based on real-time environmental data. Utilizing predictive analytics, HappySkiKit enhances user satisfaction and safety by continuously adapting to new data and user feedback.

## Components
HappySkiKit consists of two primary components:
### Happyski
- Purpose: Manages and analyzes ski resort data to provide dynamic ratings.
- Features:
  - Automated Data Management: Automates the ingestion and cleaning of ski resort data.
  - Predictive Modeling: Employs machine learning models such as XGBoost, Random Forest, and Elastic Net to predict resort ratings.
  - Continuous Learning: Facilitates model re-training and updates via user feedback and new data.
- Key Methods:
  - train(): Trains and compare all the predictive models to select the best model
  - ski_model_rate(): Use the selected model to rate new ski resorts based on their data and allow users to manually input rating scores if they do not agree with the model rating. 
  - update(): Updates the main dataset with the new ski resort data and their ratings
 
### SkiGearRecommender
- Purpose: Provides gear recommendations tailored to current weather conditions to enhance safety and comfort.
- Features:
  - Uses rule-based systems to recommend appropriate ski gear.
- Key Methods:
  - get_user_input(): Captures and processes user input to provide gear suggestions.
 
## Technical Flow
0. Data Prep: Input the resort information you collected into the "snow prediction.csv" file in the same consistent format. 
1. Initialization: Instantiate the happyski object with your dataset paths.
2. Data Processing: Automatically clean and preprocess the input data.
3. Model Training: Use happyski.train() to refine models based on historical data.
4. Rating Prediction: Apply happyski.ski_model_rate() to generate and adjust ski resort ratings.
5. Continuous Updates: Employ happyski.update() to integrate new user feedback and environmental data, enhancing prediction accuracy.
6. Gear Recommendation (Optional) : Instantiate the SkiGearRecommender object and employ get_user_input() to receive and process user input to provide gear suggestions

## Usage Instructions
```python
from happyski import happyski
from ski_gear_recommender import SkiGearRecommender

# Create happyski object with dataset paths
user_happyski = happyski("path/to/snow.csv", "path/to/prediction.csv")

# Train and compare models to select the best model
user_happyski.train()

# Rate a new ski resort (or multiple ski resorts)
user_happyski.ski_model_rate()

# Update the model with new data
user_happyski.update("path/to/new_snow.csv")

# Create a gear recommender object and get user input for recommendations
recommender = SkiGearRecommender()
recommender.get_user_input()


  
