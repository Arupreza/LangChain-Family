# -------------------------------
# 1. Import required libraries
# -------------------------------
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

# Load environment variables (e.g., Hugging Face token if required)
load_dotenv()

# -------------------------------
# 2. Define output parser
# -------------------------------
# StrOutputParser → ensures model output is parsed as plain text
parser = StrOutputParser()

# -------------------------------
# 3. Define model IDs
# -------------------------------
gemma_model = "google/gemma-2-2b-it"
llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# -------------------------------
# 4. Initialize Hugging Face model
# -------------------------------
# Using Gemma 2B IT for text generation
model = HuggingFacePipeline.from_model_id(
    model_id=gemma_model,
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,       # moderate creativity
        max_new_tokens=100,    # limit output length
        return_full_text=False # return only generated text, not the input
    )
)

# -------------------------------
# 5. Define prompts
# -------------------------------
# Prompt 1 → generate a joke about a topic
prompt_1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']   # expects a "topic" variable
)

# Prompt 2 → explain the joke that was just created
prompt_2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']    # expects the generated joke as input
)

# -------------------------------
# 6. Define sequential chain
# -------------------------------
# RunnableSequence executes steps in order:
#   1. Fill prompt_1 with topic
#   2. Generate joke with model
#   3. Parse into plain text (the joke string)
#   4. Pass joke into prompt_2
#   5. Generate explanation with model
#   6. Parse final explanation text
chain = RunnableSequence(prompt_1, model, parser, prompt_2, model, parser)

# -------------------------------
# 7. Run the chain
# -------------------------------
# Input: {"topic": "AI"}
# Step 1: Generate a joke about AI
# Step 2: Explain that joke in plain text
print(chain.invoke({'topic': 'AI'}))