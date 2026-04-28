import json

def encode(obj):
    return json.dumps(obj)

def decode(s):
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}