import tushare as ts
from dotenv import load_dotenv
import os

load_dotenv()

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')

class TushareClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TushareClient, cls).__new__(cls)
            cls._instance.pro = ts.pro_api(TUSHARE_TOKEN)
        return cls._instance

client = TushareClient()
pro = client.pro