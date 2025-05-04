from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils import Toolkit

# --- IBM Watsonx Configuration ---
project_id = "53c9bbac-1ea0-4f25-960e-b5c2e2bb6129"
api_key = "Xi6hhu-CK2M2L03CI811GAmRZbiXQ2GTzKSjVjA1GjxD"
vector_index_id = "7dfe5e16-adda-4e97-96bd-28629cd96003"
model_id = "ibm/granite-3-8b-instruct"

# Setup credentials
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key=api_key
)

# Model parameters
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "min_new_tokens": 0,
    "repetition_penalty": 1
}

# Initialize model
model = ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

# Prompt setup
prompt_input = """<|start_of_role|>system<|end_of_role|>You are a helpful assistant. Use only the information retrieved from the vector index farmingBasics to answer the following question. Answer in a complete sentence with correct capitalization and punctuation. If the answer is found, include the document name or section it came from. If the information is not available in the retrieved content, respond with "I don't know." Do not use outside knowledge or make assumptions.<|end_of_text|>"""

# FastAPI app
app = FastAPI()

# CORS middleware (to allow frontend access from browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class QuestionRequest(BaseModel):
    question: str

# Vector-based search
def proximity_search(query: str) -> str:
    try:
        api_client = APIClient(project_id=project_id, credentials=credentials)
        document_search_tool = Toolkit(api_client=api_client).get_tool("RAGQuery")
        config = {
            "vectorIndexId": vector_index_id,
            "projectId": project_id
        }
        results = document_search_tool.run(input=query, config=config)
        return results.get("output", "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# API endpoint
@app.post("/ask")
def ask_question(req: QuestionRequest):
    question = req.question
    grounding = proximity_search(question)

    if not grounding:
        return {"response": "I don't know. No relevant documents were found."}

    formatted_question = f"""<|start_of_role|>user<|end_of_role|>Use the following pieces of context to answer the question.\n\n{grounding}\n\nQuestion: {question}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"""
    prompt = f"{prompt_input}{formatted_question}"

    try:
        response = model.generate_text(prompt=prompt, guardrails=False)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)