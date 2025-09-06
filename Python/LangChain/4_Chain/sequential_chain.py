# -------------------------------
# 1. Import required libraries
# -------------------------------
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env (for Hugging Face access tokens, etc.)
load_dotenv()


# -------------------------------
# 2. Define prompt templates
# -------------------------------
# First prompt: generate a detailed report on a given topic.
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']   # "topic" will be filled at runtime
)

# Second prompt: summarize the report into 5 bullet points.
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']    # Will take the report text from prompt1 → model → parser
)


# -------------------------------
# 3. Define model IDs
# -------------------------------
gemma_model = "google/gemma-2-2b-it"
llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# -------------------------------
# 4. Initialize Hugging Face pipeline
# -------------------------------
# Similar to the first script, this creates a text generation pipeline.
llm = HuggingFacePipeline.from_model_id(
    model_id=gemma_model,
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,       # Moderate randomness
        max_new_tokens=100,    # Cap the output length
        return_full_text=False # Only return generated tokens
    )
)


# -------------------------------
# 5. Wrap the pipeline into LangChain chat model
# -------------------------------
model = ChatHuggingFace(llm=llm, model_id=gemma_model)


# -------------------------------
# 6. Define output parser
# -------------------------------
# Here we use StrOutputParser since we only need raw text strings,
# not structured objects (like Pydantic in the previous example).
parser = StrOutputParser()


# -------------------------------
# 7. Build the multi-step chain
# -------------------------------
# The pipeline works as follows:
#   Step 1: prompt1 → format request for "detailed report"
#   Step 2: model → generate report text
#   Step 3: parser → convert model output to string
#   Step 4: prompt2 → take that string, ask for 5-point summary
#   Step 5: model → generate summary text
#   Step 6: parser → clean into final string
chain = prompt1 | model | parser | prompt2 | model | parser


# -------------------------------
# 8. Run the chain
# -------------------------------
# Input topic: "Bangladesh Politics"
# Process:
#   1. Generate detailed report
#   2. Summarize into 5 points
result = chain.invoke({'topic': 'Bangladesh Politics'})

# Print the final summary
print(result)


# -------------------------------
# 9. Visualize the chain as ASCII graph
# -------------------------------
# This is useful to see the pipeline flow visually.
chain.get_graph().print_ascii()
