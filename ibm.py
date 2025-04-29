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

client = APIClient(wml_credentials)
client.set.default_project(project_id)

# --- Load Foundation Model ---
model = Model(
    model_id="ibm/granite-13b-instruct-v2",  # ✅ Supported model
    params={
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
        GenTextParamsMetaNames.TEMPERATURE: 0.7,
        GenTextParamsMetaNames.DECODING_METHOD: "sample"
    },
    credentials=wml_credentials,
    project_id=project_id
)

# --- Send a prompt ---
prompt = "Explain what a neural network is in simple terms."
response = model.generate_text(prompt)

print("Prompt:", prompt)
print("Response:", response)

