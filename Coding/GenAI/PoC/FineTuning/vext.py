import requests
import os
from dotenv import load_dotenv

load_dotenv()


VEXT_APP_API_KEY = os.getenv('VEXT_APP_API_KEY')
your_query = "What is deep learning?"


url = "https://payload.vextapp.com/hook/TAYHGKJXSQ/catch/$(Aloyplay3026)"  # Replace with the actual URL
headers = {
    "Content-Type": "application/json",
    "Apikey": "Api-Key VEXT_APP_API_KEY",  # Replace <API_KEY> with your actual API key
}
data = {
    "payload": your_query
}

response = requests.post(url, headers=headers, json=data)

print(response.status_code)
print(response.text)
