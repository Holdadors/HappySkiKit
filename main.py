from happy_ski import happyski
from ski_gear_recommend import SkiGearRecommender

if __name__ == "__main__":

    # Creating an object
    user_happyski = happyski("snow.csv", "snow prediction.csv")

    # Check train model comparison
    user_happyski.train()

    # Check prediction
    user_happyski.ski_model_rate()

    # Check update
    user_happyski.update("snow.csv")

    # Ask if user would like to receieve gear recommendation
    user_ans = input("What would you like to receive gear recommendation? (Y/N): ")
    if user_ans == "Y":
        # Ask user for snow condition to recommend gear
        recommender = SkiGearRecommender()
        recommender.get_user_input()
    else:
        pass
