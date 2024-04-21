from happy_ski import happyski
from ski_gear_recommend import SkiGearRecommender
__all__ = ['SkiResortAnalytics', 'GearRecommender']

if __name__ == "__main__":

    # Creating an object
    user_happyski = happyski("snow.csv", "snow prediction.csv")

    # Check train model comparison
    user_happyski.train()

    # Check prediction
    user_happyski.ski_model_rate()

    # Check update
    user_happyski.update("snow.csv")

    # Ask user for snow condition to recommend gear
    recommender = SkiGearRecommender()
    recommender.get_user_input()
