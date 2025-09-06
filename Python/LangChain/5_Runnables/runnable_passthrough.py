# -------------------------------
# 1. Import required libraries
# -------------------------------
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough
)

# Load environment variables (for Hugging Face API keys if needed)
load_dotenv()


# -------------------------------
# 2. Define prompt for joke generation
# -------------------------------
# Takes a topic as input and asks the model to create a joke.
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)


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
        max_new_tokens=100,    # cap output length
        return_full_text=False # return only generated text
    )
)


# -------------------------------
# 5. Define output parser
# -------------------------------
# Extracts plain text from model output
parser = StrOutputParser()


# -------------------------------
# 6. Define prompt for joke explanation
# -------------------------------
# Takes a joke as input and asks the model to explain it.
prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)


# -------------------------------
# 7. Define joke generation chain
# -------------------------------
# RunnableSequence executes in order:
#   - prompt1 → formats the topic
#   - model → generates a joke
#   - parser → extracts plain text
joke_gen_chain = RunnableSequence(prompt1, model, parser)


# -------------------------------
# 8. Define parallel chain
# -------------------------------
# RunnableParallel runs multiple branches at once:
#   - "joke": returns the joke text as-is (RunnablePassthrough)
#   - "explanation": takes the joke, runs it through prompt2 → model → parser
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})


# -------------------------------
# 9. Define final chain
# -------------------------------
# Full flow:
#   Input (topic) → joke_gen_chain → parallel_chain
#   1. Generate a joke
#   2. Run that joke into:
#       a) passthrough branch → output joke
#       b) explanation branch → explain the joke
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)


# -------------------------------
# 10. Run the chain
# -------------------------------
# Example input: topic = "cricket"
# Step 1: Generate a cricket joke
# Step 2: Return both joke and explanation
print(final_chain.invoke({'topic': 'cricket'}))