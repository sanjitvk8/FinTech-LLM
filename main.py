from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse

app = FastAPI()

# Root path
@app.get("/")
async def root():
    return {"message": "Hello Sanjana, your API is working!"}

# Input model
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

# Output model
class QueryResponse(BaseModel):
    answers: List[str]

# Dummy handler
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request_data: QueryRequest):
    dummy_answers = [
        "Dummy answer for: " + question
        for question in request_data.questions
    ]
    return JSONResponse(content={"answers": dummy_answers})
