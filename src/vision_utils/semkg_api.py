
import requests
import json

SEMKG_IMAGES_HOST = "https://files.semkg.org"
def query(query_string, token=""):
#     response = requests.post('https://vision-api.semkg.org/api/querydtd',
#                              json={"query": query_string, token: token})
    response = requests.post('https://vision.semkg.org/sparql',
                             json={"query": query_string, token: token})
    return json.loads(response.json())
