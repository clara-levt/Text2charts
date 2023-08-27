import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
import os


# Step 1: Setup front end
# Create center the titles and change the font
st.markdown("<h1 style='text-align: center; font-weight:bold; font-family:Sans-serif; padding-top: 0rem;'>Text-2-Charts</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; padding-top: 0rem;'>Creating Visualisations using Natural Language with ChatGPT </h2>", unsafe_allow_html=True)

# Create a sidebar and load the available datasets into a dictionary
if "datasets" not in st.session_state:
    datasets = {}
    # Preload datasets
    datasets["Movies"] = pd.read_csv("movies.csv")
    datasets["Housing"] = pd.read_csv("housing.csv")
    datasets["Cars"] = pd.read_csv("cars.csv")
    datasets["Colleges"] = pd.read_csv("colleges.csv")
    st.session_state["datasets"] = datasets
else:
    # use the list already loaded
    datasets = st.session_state["datasets"]

with st.sidebar:
    #Create an emply container for dataset that we can choose from
    data_container = st.empty()

    # Add facility to upload a dataset
    uploaded_file = st.file_uploader(":computer: Load a CSV file:", type="csv")
    index_no = 0 #set default dataset

    if uploaded_file:
        # Read data and add to the list of available data
        file_name = uploaded_file.name[:-4].capitalize()
        datasets[file_name] = pd.read_csv(uploaded_file)
        index_no = len(datasets) - 1  #set the newest dataset as default dataset

    #Button for choosing dataset
    chosen_dataset = data_container.radio(":bar_chart: Choose your data:", datasets.keys(), index=index_no)


#Choose LLM model
available_models = {"ChatGPT-4": "gpt-4", "ChatGPT-3.5": "gpt-3.5-turbo", "GPT-3": "text-davinci-003"}
with st.sidebar:
    st.write(":brain: Choose your model(s):")
	# Keep a dictionary of whether models are selected or not
    use_model = {}
    for model_desc,model_name in available_models.items():
        label = f"{model_desc} ({model_name})"
        key = f"key_{model_desc}"
        use_model[model_desc] = st.checkbox(label,value=True,key=key)

# Asking question area
question = st.text_area(":eyes: What would you like to visualise?", height=10)
go_btn = st.button(":robot_face: Go...")

#Display tables
tab_list = st.tabs(datasets.keys())
for dataset_num, tab in enumerate(tab_list):
    with tab:
        dataset_name = list(datasets.keys())[dataset_num]
        st.subheader(dataset_name)
        st.dataframe(datasets[dataset_name], hide_index=True)

## Adding ChatGPT Element
def run_request(question_to_ask, model_type, key):
    load_dotenv()

    # Load the OpenAI key from the environment variable
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY is None or OPENAI_API_KEY == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    openai.api_key = OPENAI_API_KEY
    if model_type == "gpt-4" or model_type == "gpt-3.5-turbo":
        # Run ChatGPT API
        response = openai.ChatCompletion.create(
            model=model_type,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates Python code."},
                {"role": "user", "content": question_to_ask}
            ]
        )
        res = response["choices"][0]["message"]["content"]
    else:
        response = openai.Completion.create(
            engine=model_type,
            prompt=question_to_ask,
            temperature=0.7,  # You can adjust these parameters as needed
            max_tokens=500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["plt.show()"]
        )
        res = response["choices"][0]["text"]
    return res

# Example usage
question = "How do I sort a list in Python?"
model = "gpt-3.5-turbo"
result = run_request(question, model, "your_api_key_here")
print(result)
