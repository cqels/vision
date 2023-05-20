
import requests
import json

SEMKG_IMAGES_HOST = "https://files.semkg.org"
              

def query(query_string, token=""):
#     response = requests.post('https://vision-api.semkg.org/api/querydtd',
#                              json={"query": query_string, token: token})
#     response = requests.post('https://vision.semkg.org/sparql',
#                              json={"query": query_string, token: token})
# temp api
      response = requests.post('https://vision-api.semkg.org/api/sparql_temp',
                               json={"query": query_string, token: token})
      
      _data=json.loads(response.json())
      data=[]
      for result in _data.results.bindings:
            tmp={}
            for key in result:
                  tmp[key]=result[key].value
            data.append(tmp)
      return data
      #return json.loads(response.json())
