import requests

while True:
    query = input("Enter the query (type 'exit' or 'quit' to stop): ").strip()
    
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    try:
        response = requests.post("http://localhost:8000/query", 
                                 json={"query": query})
        print("Status:", response.status_code)

        data = response.json()
        if "response" in data:
            print("Answer:", data["response"])
        else:
            print("Full Response:", data)

    except Exception as e:
        print("Error:", e)
        print("Raw Response:", response.text if response else "No response")
