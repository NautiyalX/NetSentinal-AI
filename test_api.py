
import json
import urllib.request
import urllib.error

url = "http://127.0.0.1:10000/api/predict"

payload = {
    "service": "http",
    "flag": "SF",
    "count": 10,
    "dst_host_srv_count": 10,
    "dst_host_diff_srv_rate": 1.5,
    "dst_host_same_src_port_rate": 0.5,
    "dst_host_same_srv_rate": 0.8,
    "same_srv_rate": 0.8,
    "serror_rate": 0.1,
    "last_flag": 10,
    "logged_in": 0
}

request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(request) as response:
        print("HTTP status:", response.status)
        print(response.read().decode())

except urllib.error.HTTPError as error:
    print("HTTP status:", error.code)
    print(error.read().decode())
