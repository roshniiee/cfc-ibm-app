from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames
from ibm_watson_machine_learning import APIClient

# --- Configuration ---
api_key = "UWSEESFgNCr3MjkErE_axi44qhF-gnAyyEaXRBBupt2Y"
project_id = "5f234a65-389a-4d14-b5ae-df50c8bfab47"
region = "eu-de"
url = "https://eu-de.ml.cloud.ibm.com"

# --- Set up WML client ---
wml_credentials = {
    "apikey": api_key,
    "url": url,
}

try:
    client = APIClient(wml_credentials)
    client.set.default_project(project_id)

    # --- Load Foundation Model (e.g., Granite 13B Chat) ---
    model = Model(
        model_id="meta-llama/llama-2-13b-chat",  # Corrected model
        params={
            GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
            GenTextParamsMetaNames.TEMPERATURE: 0.5,
            GenTextParamsMetaNames.DECODING_METHOD: "greedy"
        },
        credentials=wml_credentials,
        project_id=project_id
    )

    # --- Send a prompt ---
    response = model.generate_text("give me basic requirements to be a farmer.")
    print(response)

except Exception as e:
    print("Error during setup or model invocation:", e)
