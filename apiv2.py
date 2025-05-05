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

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- IBM Watsonx Configuration ---
project_id = "53c9bbac-1ea0-4f25-960e-b5c2e2bb6129"
api_key = "Xi6hhu-CK2M2L03CI811GAmRZbiXQ2GTzKSjVjA1GjxD" # Consider using environment variables for sensitive data
vector_index_id = "7dfe5e16-adda-4e97-96bd-28629cd96003"
model_id = "ibm/granite-3-8b-instruct" # NOTE: Changed model ID based on common availability - verify this is correct for your project/region

district_value="Pune"

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
    logger.error(f"Failed to initialize model: {e}")
    # Depending on your application structure, you might want to exit or raise here
    raise RuntimeError(f"Failed to initialize Watsonx model: {e}")


# Prompt setup (System Prompt)
# Ensure this system prompt structure aligns with Granite v2 instructions if you changed the model ID
system_prompt = """<|system|>You are a helpful assistant. Use only the information retrieved from the vector index farmingBasics to answer the following question.
Answer in a complete sentence with correct capitalization and punctuation.
If the answer is found, try to include the document name or source it came from based on the provided context.
If the information is not available in the retrieved content, respond with "I don't know based on the provided information."
Do not use outside knowledge or make assumptions.

*FORMATTING INSTRUCTION:* If the retrieved context contains data presented in a table, and that table is relevant and necessary to answer the question, *reproduce the relevant tabular data in your answer using GitHub Flavored Markdown table format.* Ensure the Markdown table is correctly formatted with headers (| Header 1 | Header 2 |), separator lines (|---|---|), and data rows (| Cell 1 | Cell 2 |). Present any non-tabular parts of the answer as normal text paragraphs before or after the Markdown table as appropriate.
<|end_system|>
"""

# FastAPI app
app = FastAPI()

# CORS middleware (to allow frontend access from browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # IMPORTANT: Restrict this in production environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class QuestionRequest(BaseModel):
    question: str
    
class DistrictInput(BaseModel):
    district: str

# Vector-based search - Modified to return detailed results
def proximity_search(query: str) -> dict: # Return the full dictionary
    """
    Performs a proximity search using the Watsonx RAGQuery tool.

    Args:
        query: The user's question.

    Returns:
        A dictionary containing the search results from the RAG tool.
        Expected keys might include 'output' (formatted string of results),
        'context' (list of document chunks with metadata), etc.
        Returns an empty dict if an error occurs.
    """
    try:
        logger.info(f"Performing proximity search for query: '{query}'")
        # It's often better to initialize the client within the function if it's not needed globally
        # or if credentials/project_id could change per request (though not the case here).
        api_client = APIClient(project_id=project_id, credentials=credentials)
        document_search_tool = Toolkit(api_client=api_client).get_tool("RAGQuery")
        config = {
            "vectorIndexId": vector_index_id,
            "projectId": project_id
            # You might add other parameters here like 'level': 'document' or 'chunk'
            # or 'retrieve_only': True if you only want retrieval without synthesis
            # 'max_results': N  (if supported by the tool)
        }
        results = document_search_tool.run(input=query, config=config)
        logger.info(f"Proximity search raw results: {results}") # Log the raw results for inspection

        # --- IMPORTANT: Inspect the 'results' structure ---
        # The exact structure of 'results' depends on the RAGQuery tool version and config.
        # Common structures include:
        # results = {
        #   'output': 'Formatted string...',
        #   'context': {
        #     'documents': [
        #       {'text': 'chunk content', 'metadata': {'source': 'doc_name.pdf', 'score': 0.85, ...}},
        #       ...
        #     ]
        #   }
        # }
        # Verify this structure by printing/logging results during a test run.

        return results # Return the whole dictionary

    except Exception as e:
        logger.error(f"Proximity search error for query '{query}': {str(e)}", exc_info=True)
        # Return empty dict or raise specific exception for the endpoint to catch
        return {}
        # Or: raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


# API endpoint - Modified to return detailed response
@app.post("/ask2")
def ask_question(req: QuestionRequest):
    question = req.question
    logger.info(f"Received question: '{question}'")
    
    print(f"\nAttempting to generate forecast for: {district_value}")

    # 1. Perform Vector Search
    search_results = proximity_search(question)

    if not search_results or not search_results.get("output"): # Check if search returned anything useful
        logger.warning(f"No relevant documents found for question: '{question}'")
        # Return a structured response even for "I don't know"
        return {
            "question": question,
            "response_text": "I don't know. No relevant documents were found in the vector index.",
            "retrieved_context": [],
            "generation_details": None,
            "search_error": "No results from vector search" if not search_results else None
        }

    # Extract the formatted context string needed for the prompt
    # Assuming the 'output' key contains the concatenated context string
    grounding_context = search_results.get("output", "")
    if not grounding_context:
         logger.warning(f"Search results dictionary present, but 'output' key is empty or missing for question: '{question}'")
         # Decide how to handle this - maybe proceed without context or return error
         return {
            "question": question,
            "response_text": "I don't know. Search results were found but context could not be formatted.",
            "retrieved_context": search_results.get("context", {}).get("documents", []), # Still return raw context if available
            "generation_details": None,
            "search_error": "Search results missing 'output' key"
        }
        
    weather_context = format_weather_forecast()    
        
    # --- Combine Contexts ---
    combined_context = ""
    if grounding_context:
        combined_context += f"Retrieved Farming Information:\n{grounding_context}\n\n" # Added header
    if weather_context:
        combined_context += f"{weather_context}\n\n" # Header is added within format_weather_forecast

    # Handle case where *neither* search yielded results
    if not combined_context:
         logger.warning(f"No context found from Vector Search or Weather Data for: '{question}'")
         return {
             "question": question,
             "response_text": "I don't know. No relevant farming documents or weather forecast data were found.",
             "retrieved_context": retrieved_documents,
             "generation_details": None,
             "search_error": search_error if search_error else "No context found from any source"
         }

    # Extract detailed retrieved documents (adjust key names based on actual RAG tool output)
    retrieved_documents = search_results.get("context", {}).get("documents", [])


    # 2. Format the Prompt for the Model
    # Using the structure typically expected by Granite Instruct models
    # <|system|>...<|end_system|>
    # <|user|>Context: ... \n\n Question: ...<|end_user|>
    # <|assistant|>
    formatted_prompt = f"""{system_prompt}
<|user|>Context:
{combined_context.strip()}

Question: {question}<|end_user|>
<|assistant|>"""

    logger.info(f"Formatted prompt for model:\n{formatted_prompt}")


    # 3. Generate Response using the Model
    try:
        logger.info("Sending prompt to model for generation...")
        # Use generate() instead of generate_text()
        model_response = model.generate(prompt=formatted_prompt, guardrails=False) # Pass guardrails here if needed by generate()
        logger.info(f"Raw model response: {model_response}")

        # --- IMPORTANT: Inspect the 'model_response' structure ---
        # The structure depends on the Watsonx API version. It's often nested.
        # Example structure:
        # model_response = {
        #     "model_id": "...",
        #     "created_at": "...",
        #     "results": [
        #         {
        #             "generated_text": "The generated answer text.",
        #             "generated_token_count": 75,
        #             "input_token_count": 350,
        #             "stop_reason": "eos_token" | "max_tokens" | ...
        #             # Potentially other fields like logprobs, etc.
        #         }
        #     ],
        #     # Maybe system-level fields
        # }
        # Extract the relevant parts carefully.

        if not model_response or not model_response.get("results"):
             raise ValueError("Model response is empty or missing 'results' key.")

        # Safely extract details
        result_data = model_response.get("results", [{}])[0] # Get the first result dictionary
        generated_text_raw = result_data.get("generated_text", "Error: Could not extract text from model response.")

        # --- ADD THIS LINE ---
        # Remove the unwanted token from the end of the response
        # generated_text = generated_text_raw.rstrip("<|end_assistant|>").strip() # Use rstrip if it only appears at the end, strip() removes leading/trailing whitespace
        # Or use replace if it might appear elsewhere (less likely for an end token)
        generated_text = generated_text_raw.replace("<|end_assistant|>", "").strip()
        generation_details = {
            "stop_reason": result_data.get("stop_reason"),
            "generated_token_count": result_data.get("generated_token_count"),
            "input_token_count": result_data.get("input_token_count"),
             # Add any other fields you find useful from result_data
        }
        model_id_used = model_response.get("model_id")
        if model_id_used:
            generation_details["model_id_used"] = model_id_used


        # 4. Return Detailed Response
        return {
            "question": question,
            "response_text": generated_text, # Clean up whitespace
            "retrieved_context": retrieved_documents, # List of document chunks/metadata
            "generation_details": generation_details, # Dictionary of model metadata
            "search_error": None
        }

    except Exception as e:
        logger.error(f"Model generation error for question '{question}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

def format_weather_forecast():
    days_to_include: int = 15
    required_cols = ['Temperature', 'Humidity', 'Rainfall']
    
    print(f"\nAttempting to generate forecast for: {district_value}")
    try:
        # 4. Call the imported function with the required arguments
        forecast_df = generate_district_forecast(
            district_name=district_value,
            wml_api_key=api_key
            # If you added more configurable arguments to the function, pass them here:
            # e.g., project_id='your-specific-project-id'
        )

        # 5. Process the results
        if forecast_df is not None and isinstance(forecast_df, pd.DataFrame):
            print("\nForecast Generated Successfully:")
            print(forecast_df)
 
        elif forecast_df is None:
            print(f"\nForecast generation failed for {DISTRICT}. Check logs from the function.")
        else:
            # Should ideally not happen if function returns DataFrame or None
             print(f"\nForecast generation returned an unexpected type: {type(forecast_df)}")
             
        if not isinstance(forecast_df.index, pd.DatetimeIndex):
             # If the date is a regular column (e.g., 'Date'), set it as index
             if 'Date' in forecast_df.columns:
                 forecast_df = forecast_df.set_index(pd.to_datetime(forecast_df['Date']))
             elif isinstance(forecast_df.index, pd.Index) and forecast_df.index.dtype == 'object':
                 # Try converting an existing object index
                 forecast_df.index = pd.to_datetime(forecast_df.index)
             else:
                  logger.warning("DataFrame index is not a DatetimeIndex and no 'Date' column found.")
                  return "" # Cannot proceed without dates


        # Select the forecast period and relevant columns
        forecast_subset = forecast_df.head(days_to_include)[required_cols].copy()

        # --- Format the Output String ---
        # Option 1: Simple List Format
        # forecast_lines = ["Upcoming Weather Forecast:"]
        # for date, row in forecast_subset.iterrows():
        #     forecast_lines.append(
        #         f"- {date.strftime('%Y-%m-%d')}: "
        #         f"Temp: {row['Temperature']:.1f}°C, " # Round for readability
        #         f"Humidity: {row['Humidity']:.1f}%, "
        #         f"Rainfall: {row['Rainfall']:.2f}mm"
        #     )
        # formatted_string = "\n".join(forecast_lines)

        # Option 2: More structured Table-like Format using io.StringIO
        # (Requires pandas >= 1.0.0 for to_markdown, alternatively use to_string)
        forecast_subset.index = forecast_subset.index.strftime('%Y-%m-%d') # Format date index for display
        forecast_subset.rename(columns={
            'Temperature': 'Temp (°C)',
            'Humidity': 'Humidity (%)',
            'Rainfall': 'Rainfall (mm)'
        }, inplace=True)
        # Round values
        forecast_subset['Temp (°C)'] = forecast_subset['Temp (°C)'].round(1)
        forecast_subset['Humidity (%)'] = forecast_subset['Humidity (%)'].round(1)
        forecast_subset['Rainfall (mm)'] = forecast_subset['Rainfall (mm)'].round(2)

        output = io.StringIO()
        # forecast_subset.to_markdown(output, tablefmt="grid") # Use if markdown is preferred and supported
        forecast_subset.to_string(output) # Standard string representation
        formatted_string = f"Upcoming Weather Forecast (Next {len(forecast_subset)} Days):\n{output.getvalue()}"
        output.close()

        logger.info(f"Formatted weather forecast for {len(forecast_subset)} days.")
        logger.info(f"Formatted String {formatted_string}")
        return formatted_string

    except Exception as e:
        # Catch any unexpected errors during the function call itself
        print(f"\nAn error occurred while calling generate_district_forecast: {e}")
        import traceback
        traceback.print_exc()

@app.post("/set_district")
def set_district(data: DistrictInput):
    global district_value
    district_value = data.district
    return {"message": f"District set to '{district_value}'"}
    
if __name__ == "_main_":
    import uvicorn
    # Note: reload=True is great for development but should be False in production
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)