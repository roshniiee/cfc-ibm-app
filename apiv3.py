from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils import Toolkit
from generateforecast import generate_district_forecast
import logging # Added for better debugging
import pandas as pd
import io
from typing import Optional # Import Optional for clarity

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- IBM Watsonx Configuration ---
project_id = "53c9bbac-1ea0-4f25-960e-b5c2e2bb6129"
api_key = "Xi6hhu-CK2M2L03CI811GAmRZbiXQ2GTzKSjVjA1GjxD" # Use environment variables
vector_index_id = "7dfe5e16-adda-4e97-96bd-28629cd96003"
model_id = "ibm/granite-3-8b-instruct" # Verify this model ID

# --- Global Variables ---
# These hold the state set by /set_district
# Initialize district_value (or handle cases where /ask2 is called before /set_district)
district_value: Optional[str] = "Pune" # Default or None if you want to force setting it first
weather_string: Optional[str] = "" # Initialize weather string

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
try:
    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )
    logger.info(f"Successfully initialized model: {model_id}")
except Exception as e:
    logger.error(f"Failed to initialize model: {e}", exc_info=True)
    raise RuntimeError(f"Failed to initialize Watsonx model: {e}")


# Prompt setup (System Prompt) - Make sure it aligns with your model
system_prompt = """<|system|>You are a helpful agricultural assistant. Use only the information retrieved from the vector index farmingBasics, the specified district context, and the provided weather forecast to answer the following question regarding farming.
Answer in a complete sentence with correct capitalization and punctuation.
If the answer is found, try to include the document name or source it came from based on the provided context.
If the information is not available in the retrieved content or forecast, respond with "I don't know based on the provided information."
Do not use outside knowledge or make assumptions. The query is specific to the provided district.

*FORMATTING INSTRUCTION:* If the retrieved context contains data presented in a table, and that table is relevant and necessary to answer the question, *reproduce the relevant tabular data in your answer using GitHub Flavored Markdown table format.* Ensure the Markdown table is correctly formatted with headers (| Header 1 | Header 2 |), separator lines (|---|---|), and data rows (| Cell 1 | Cell 2 |). Present any non-tabular parts of the answer as normal text paragraphs before or after the Markdown table as appropriate.
<|end_system|>
""" # Added mention of district and weather

# FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input models
class QuestionRequest(BaseModel):
    question: str

class DistrictInput(BaseModel):
    district: str

# Vector-based search (remains the same)
def proximity_search(query: str) -> dict:
    """Performs proximity search using Watsonx RAGQuery tool."""
    try:
        logger.info(f"Performing proximity search for query: '{query}'")
        api_client = APIClient(project_id=project_id, credentials=credentials)
        document_search_tool = Toolkit(api_client=api_client).get_tool("RAGQuery")
        config = {
            "vectorIndexId": vector_index_id,
            "projectId": project_id
        }
        results = document_search_tool.run(input=query, config=config)
        logger.info(f"Proximity search raw results (truncated): {str(results)[:500]}...") # Log truncated results
        return results
    except Exception as e:
        logger.error(f"Proximity search error for query '{query}': {str(e)}", exc_info=True)
        return {}


# API endpoint - Modified to include district_value
@app.post("/ask2")
def ask_question(req: QuestionRequest):
    original_question = req.question
    logger.info(f"Received original question: '{original_question}'")

    # --- Get District and Weather Context (from global state) ---
    current_district = district_value # Access the global variable
    current_weather_context = weather_string # Access the global variable

    if not current_district:
        # Handle case where district hasn't been set - return error or use default
        logger.error("District value has not been set. Please call /set_district first.")
        raise HTTPException(status_code=400, detail="District context not set. Please call /set_district first.")
        # Or, if you want to proceed with a default/no district:
        # current_district = "Default District" # Or handle differently

    logger.info(f"Using District Context: '{current_district}'")
    logger.info(f"Using Weather Context (available: {bool(current_weather_context)})")

    # --- 1. Preprocess Question with District Context ---
    # Add district context to the question for vector search
    processed_question = f"Regarding farming in {current_district} district: {original_question}"
    logger.info(f"Processed question for vector search: '{processed_question}'")

    # --- 2. Perform Vector Search (using processed_question) ---
    search_results = proximity_search(processed_question)

    retrieved_documents = []
    grounding_context = ""
    search_error = None

    if not search_results or not search_results.get("output"):
        logger.warning(f"No relevant documents found via vector search for: '{processed_question}'")
        search_error = "No results from vector search"
    else:
        grounding_context = search_results.get("output", "")
        retrieved_documents = search_results.get("context", {}).get("documents", [])
        if not grounding_context:
            logger.warning(f"Vector search results present, but 'output' is empty for: '{processed_question}'")
            search_error = "Search results missing 'output' key"

    # --- 3. Combine Contexts for LLM Prompt ---
    combined_context_parts = []

    # Add District Context explicitly for the LLM
    combined_context_parts.append(f"District Context: {current_district}")

    # Add Retrieved Documents Context
    if grounding_context:
        combined_context_parts.append(f"Retrieved Farming Information:\n{grounding_context}")

    # Add Weather Forecast Context
    if current_weather_context:
        combined_context_parts.append(current_weather_context) # Header is already in weather_string

    combined_context = "\n\n".join(combined_context_parts) # Join parts with double newline

    # Handle case where no context is available at all (unlikely now with district added)
    if not grounding_context and not current_weather_context:
        logger.warning(f"Only district context available for: '{processed_question}'")
        # Decide how to proceed - maybe rely solely on the model's internal knowledge + district?
        # Or return a specific message:
        # return { ... "response_text": f"I have context for {current_district}, but no specific documents or weather forecast were found for your question." ...}

    # --- 4. Format the Prompt for the Model ---
    # Use combined_context and the *original_question* (or processed_question, debatable)
    # Using original_question here might feel more natural in the prompt structure,
    # as the district context is already clearly laid out above it.
    formatted_prompt = f"""{system_prompt}
<|user|>Context:
{combined_context.strip()}

Question: {original_question}<|end_user|>
<|assistant|>"""
    # Alternatively, if you want the question itself to reiterate the district:
    # formatted_prompt = f"""{system_prompt}
    # <|user|>Context:
    # {combined_context.strip()}
    #
    # Question: {processed_question}<|end_user|>
    # <|assistant|>"""

    logger.info(f"Formatted prompt for model (truncated):\n{formatted_prompt[:1000]}...")

    # --- 5. Generate Response using the Model ---
    try:
        logger.info("Sending prompt to model for generation...")
        model_response = model.generate(prompt=formatted_prompt, guardrails=False)
        logger.info(f"Raw model response: {model_response}")

        if not model_response or not model_response.get("results"):
             raise ValueError("Model response is empty or missing 'results' key.")

        result_data = model_response.get("results", [{}])[0]
        generated_text_raw = result_data.get("generated_text", "Error: Could not extract text from model response.")
        generated_text = generated_text_raw.replace("<|end_assistant|>", "").strip()

        generation_details = {
             "stop_reason": result_data.get("stop_reason"),
             "generated_token_count": result_data.get("generated_token_count"),
             "input_token_count": result_data.get("input_token_count"),
             "model_id_used": model_response.get("model_id")
        }

        # --- 6. Return Detailed Response ---
        return {
             "original_question": original_question,
             "processed_question_for_search": processed_question, # Show what was searched
             "district_context_provided": current_district,
             "weather_forecast_provided": bool(current_weather_context),
             "response_text": generated_text,
             "retrieved_context": retrieved_documents,
             "generation_details": generation_details,
             "search_error": search_error
        }

    except Exception as e:
        logger.error(f"Model generation error for question '{original_question}' (District: {current_district}): {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

# Weather forecast formatting function (remains mostly the same, uses global district_value)
def format_weather_forecast():
    """Generates forecast using global district_value and formats it into global weather_string."""
    global weather_string # Declare intent to modify global variable
    days_to_include: int = 15
    required_cols = ['Temperature', 'Humidity', 'Rainfall']
    current_district = district_value # Use the global value

    if not current_district:
        logger.error("Cannot format weather forecast: district_value is not set.")
        weather_string = "" # Clear any previous forecast
        return # Exit if no district is set

    logger.info(f"Attempting to generate forecast for: {current_district}")
    try:
        forecast_df = generate_district_forecast(
            district_name=current_district,
            wml_api_key=api_key
        )

        if forecast_df is None or not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty:
            logger.warning(f"Forecast generation failed or returned empty DataFrame for {current_district}.")
            weather_string = "" # Ensure empty if failed
            return # Exit if forecast failed

        logger.info(f"Forecast DataFrame received for {current_district}.")

        # --- DataFrame processing and formatting (same as before) ---
        if not isinstance(forecast_df.index, pd.DatetimeIndex):
             if 'Date' in forecast_df.columns:
                 forecast_df = forecast_df.set_index(pd.to_datetime(forecast_df['Date']))
             elif isinstance(forecast_df.index, pd.Index) and forecast_df.index.dtype == 'object':
                 forecast_df.index = pd.to_datetime(forecast_df.index)
             else:
                  logger.warning("Forecast DataFrame index is not DatetimeIndex and no 'Date' column found.")
                  weather_string = ""
                  return

        # Ensure required columns are present *after* potential index setting
        if not all(col in forecast_df.columns for col in required_cols):
             logger.warning(f"Forecast DataFrame for {current_district} missing required columns after processing.")
             weather_string = ""
             return

        forecast_subset = forecast_df.head(days_to_include)[required_cols].copy()

        forecast_subset.index = forecast_subset.index.strftime('%Y-%m-%d')
        forecast_subset.rename(columns={
             'Temperature': 'Temp (°C)', 'Humidity': 'Humidity (%)', 'Rainfall': 'Rainfall (mm)'
        }, inplace=True)
        forecast_subset['Temp (°C)'] = forecast_subset['Temp (°C)'].round(1)
        forecast_subset['Humidity (%)'] = forecast_subset['Humidity (%)'].round(1)
        forecast_subset['Rainfall (mm)'] = forecast_subset['Rainfall (mm)'].round(2)

        output = io.StringIO()
        forecast_subset.to_string(output)
        formatted_string = f"Upcoming Weather Forecast for {current_district} (Next {len(forecast_subset)} Days):\n{output.getvalue()}" # Added district to header
        output.close()

        logger.info(f"Formatted weather forecast for {len(forecast_subset)} days for {current_district}.")
        weather_string = formatted_string # Update the global variable

    except Exception as e:
        logger.error(f"An error occurred during weather forecast generation/formatting for {current_district}: {e}", exc_info=True)
        weather_string = "" # Clear weather string on error

# Endpoint to set district and trigger forecast generation
@app.post("/set_district")
def set_district(data: DistrictInput):
    global district_value # Declare intent to modify global
    old_district = district_value
    district_value = data.district
    logger.info(f"District changed from '{old_district}' to '{district_value}'. Triggering forecast generation.")
    format_weather_forecast() # Generate forecast for the new district
    return {"message": f"District set to '{district_value}'. Weather forecast updated."}

# Main execution block (remains the same)
if __name__ == "__main__":
    import uvicorn
    # Generate initial forecast for default district on startup? Optional.
    # logger.info("Generating initial forecast on startup...")
    # format_weather_forecast()
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) # Assuming filename is api.py