# Script to exercise the locally running census income API.
import requests

BASE_URL = "http://127.0.0.1:8000"

record = {
    "age": 52,
    "workclass": "Self-emp-inc",
    "fnlwgt": 209642,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States",
}


def main():
    """Call GET / and POST /predict and print status codes and results."""
    response = requests.get(f"{BASE_URL}/", timeout=10)
    print("GET /")
    print(f"Status Code: {response.status_code}")
    print(f"Result: {response.json()}")

    response = requests.post(f"{BASE_URL}/predict", json=record, timeout=10)
    print("POST /predict")
    print(f"Status Code: {response.status_code}")
    print(f"Result: {response.json()}")


if __name__ == "__main__":
    main()
