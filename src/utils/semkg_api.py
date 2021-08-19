import requests
import json


def query(query_string, token=""):
    response = requests.post('https://vision-api.semkg.org/api/querydtd',
                             json={"query": query_string, token: token})
    return json.loads(response.json())
