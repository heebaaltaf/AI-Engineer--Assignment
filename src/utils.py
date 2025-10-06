import requests
import os
from dotenv import load_dotenv

load_dotenv()

# def get_weather(city: str) -> str:
#     api_key = os.getenv("OPENWEATHER_API_KEY")
#     if not api_key:
#         return "API key not found"
#     url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         data = response.json()
#         temp = data['main']['temp']
#         desc = data['weather'][0]['description']
#         return f"Temperature: {temp}°C, Description: {desc}"
#     except Exception as e:
#         return f"Error fetching weather: {str(e)}"

import os
import requests

def get_weather(city: str) -> str:
    """Fetch live weather data for a given city using OpenWeatherMap API."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "API key not found. Please set OPENWEATHER_API_KEY in your environment."

    city = city.strip().replace(" ", "+")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("cod") != 200:
            return f"Could not find weather data for '{city.replace('+', ' ')}'."

        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"].capitalize()
        humidity = data["main"].get("humidity", "N/A")
        wind = data["wind"].get("speed", "N/A")

        return (
            f"Temperature: {temp}°C\n"
            f"Condition: {desc}\n"
            f"Humidity: {humidity}%\n"
            f"Wind speed: {wind} m/s"
        )

    except requests.exceptions.RequestException as e:
        return f"Network error while fetching weather: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
