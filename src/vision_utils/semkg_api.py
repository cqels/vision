
import requests
import json

SEMKG_IMAGES_HOST = "https://files.semkg.org"
              

def query(query_string, token=""):
      response = requests.get('https://vision.semkg.org/sparql',
                             json={"query": query_string, token: token})
      _data=response.json()
      data=[]
      # pprint(_data)
      for result in _data['results']['bindings']:
            tmp={}
            for key in result.keys():
                  tmp[key]=result[key]['value']
            data.append(tmp)
      return data