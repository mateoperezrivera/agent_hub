import random
from typing import Annotated
from semantic_kernel.functions import kernel_function

class WeatherPlugin:
    """A simple weather plugin that returns mock weather data"""
    
    @kernel_function(
        name="get_weather",
        description="Get the current weather for a specified location"
    )
    def get_weather(
        self,
        location: Annotated[str, "The city and state, e.g., Seattle, WA"] = "Seattle, WA"
    ) -> str:
        """Get mock weather data for a location"""
        
        # Mock weather conditions
        conditions = ["sunny", "partly cloudy", "cloudy", "rainy", "snowy"]
        temperature = random.randint(30, 85)
        condition = random.choice(conditions)
        humidity = random.randint(30, 80)
        wind_speed = random.randint(5, 25)
        
        weather_report = (
            f"Weather in {location}:\n"
            f"🌡️ Temperature: {temperature}°F\n"
            f"☁️ Conditions: {condition.capitalize()}\n"
            f"💧 Humidity: {humidity}%\n"
            f"💨 Wind Speed: {wind_speed} mph"
        )
        
        return weather_report
