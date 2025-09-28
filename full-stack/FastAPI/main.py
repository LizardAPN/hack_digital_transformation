from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

@app.get("/api")
def read_api_root():
    return {"message": "Welcome to the API"}

@app.get("/api/items")
def read_items():
    return {"items": ["item1", "item2", "item3"]}

@app.get("/api/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "name": f"Item {item_id}"}
