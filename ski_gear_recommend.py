
'''
Yaoyao Zheng
DS5010 project
'''
class SkiGearRecommender:
    def __init__(self):
        # Gear maps based on temperature and weather conditions
        self.gear_by_temperature = {
            'cold': ['insulated jacket', 'gloves', 'wool hat', 'scarf'],
            'moderate': ['light jacket', 'liner gloves'],
            'warm': ['sweatshirt']
        }
        self.gear_by_condition = {
            'snow': ['snow goggles', 'snowboard boots', 'snow jacket'],
            'clear': [],
            'windy': ['windbreaker'],
            'rain': ['waterproof jacket', 'waterproof pants']
        }
        # Lens recommendations based on UV index
        self.lens_by_uv = {
            0: "Clear lens",
            1: "Clear lens",
            2: "Yellow lens for low light",
            3: "Gold lens for moderate light",
            4: "Gold lens for moderate light",
            5: "Amber lens for bright light",
            6: "Amber lens for bright light",
            7: "Dark brown lens for bright light",
            8: "Dark brown lens for very bright light",
            9: "Dark brown lens for very bright light",
            10: "Black lens for intense light",
            11: "Black lens for intense light"
        }

    def recommend_gear(self, temperature, condition):
        # Determine temperature category
        if temperature <= 32:  # temperatures in Fahrenheit
            temp_category = 'cold'
        elif 32 < temperature <= 50:
            temp_category = 'moderate'
        else:
            temp_category = 'warm'

        # Get gear recommendations
        recommended_gear = set(self.gear_by_temperature[temp_category] + self.gear_by_condition[condition])
        return recommended_gear

    def recommend_lens(self, uv_index):
        # Normalize UV index to be within the expected range (0-11)
        uv_index = max(0, min(uv_index, 11))
        return self.lens_by_uv[uv_index]

    def get_user_input(self):
        # Get temperature input
        user_input = "Y"
        while user_input == "Y":
            temperature = float(input("Enter the temperature (in Fahrenheit): "))
            # Get condition input
            condition = input("Enter the current weather condition (options: snow, clear, windy, rain): ").lower()
            # Get UV index input
            uv_index = int(input("Enter the UV index (0-11): "))

            # Generate recommendations
            gear = self.recommend_gear(temperature, condition)
            lens = self.recommend_lens(uv_index)

            print("\nRecommended Gear:", gear)
            print("Recommended Lens:", lens)

            user_input = input("Do you want to continue (Y/N): ")
            if user_input == "N":
                break

