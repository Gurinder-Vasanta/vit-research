from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")

db = client['TEST_DA_DB']
collection = db['yoooo']

collection.insert_one({'a':23})