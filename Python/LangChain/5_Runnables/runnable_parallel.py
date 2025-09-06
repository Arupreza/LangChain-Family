# -------------------------------
# 1. Import required libraries
# -------------------------------
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence

# Load environment variables (e.g., Hugging Face token if required)
load_dotenv()

# -------------------------------
# 2. Define output parser
# -------------------------------
# StrOutputParser is used to clean and return plain string output
parser = StrOutputParser()

# -------------------------------
# 3. Define prompts
# -------------------------------
# Prompt 1 → generate a tweet (short, concise post)
prompt_1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']   # expects a variable "topic"
)

# Prompt 2 → generate a LinkedIn post (longer, professional post)
prompt_2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

# -------------------------------
# 4. Define model IDs
# -------------------------------
gemma_model = "google/gemma-2-2b-it"
llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# -------------------------------
# 5. Initialize Hugging Face model
# -------------------------------
# Here we are using the Gemma model for generation
model = HuggingFacePipeline.from_model_id(
    model_id=gemma_model,
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,       # controls creativity
        max_new_tokens=100,    # cap output length
        return_full_text=False # return only generated text, not the input
    )
)

# -------------------------------
# 6. Define parallel chain
# -------------------------------
# RunnableParallel allows executing multiple chains simultaneously.
# Here we create two parallel branches:
#   - "tweet": runs prompt_1 → model → parser
#   - "linkedin": runs prompt_2 → model → parser
parllel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt_1, model, parser),
    'linkedin': RunnableSequence(prompt_2, model, parser)
})

# -------------------------------
# 7. Run the chain
# -------------------------------
# Input: {"topic": "AI"}
# The same input is passed to both branches.
#   - The "tweet" branch generates a short tweet about AI
#   - The "linkedin" branch generates a professional LinkedIn post about AI
result = parllel_chain.invoke({'topic': 'AI'})

# -------------------------------
# 8. Print results
# -------------------------------
# Access the dictionary keys "tweet" and "linkedin"
print(result['tweet'])     # short social-media-friendly tweet
print(result['linkedin'])  # longer, professional LinkedIn post