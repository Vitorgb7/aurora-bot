import os
from dotenv import load_dotenv

load_dotenv()

cache = {}

def get_cached_response(key: str):
    return cache.get(key)

def set_cached_response(key: str, value: str):
    cache[key] = value
