# -------------------------------
# 1. Import required libraries
# -------------------------------
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import (
    RunnableSequence,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel
)

# Load environment variables (Hugging Face tokens etc.)
load_dotenv()


# -------------------------------
# 2. Helper function
# -------------------------------
# Simple Python function that counts the number of words in a text
def word_count(text):
    return len(text.split())


# -------------------------------
# 3. Define prompt
# -------------------------------
# Prompt template for generating a joke given a topic
prompt = PromptTemplate(
    template='Write a joke about {topic}',
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
# Using Gemma 2B IT for text generation
model = HuggingFacePipeline.from_model_id(
    model_id=gemma_model,
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,       # moderate randomness
        max_new_tokens=100,    # limit joke length
        return_full_text=False # return only generated text
    )
)


# -------------------------------
# 6. Define output parser
# -------------------------------
# Converts model output to a plain string
parser = StrOutputParser()


# -------------------------------
# 7. Define joke generation chain
# -------------------------------
# RunnableSequence executes in order:
#   1. prompt → fill with topic
#   2. model → generate joke
#   3. parser → extract string
joke_gen_chain = RunnableSequence(prompt, model, parser)


# -------------------------------
# 8. Define parallel chain
# -------------------------------
# RunnableParallel executes multiple tasks at once
# Input: generated joke string
# Output: dictionary with two keys:
#   - "joke": the joke itself (RunnablePassthrough just forwards input)
#   - "word_count": applies word_count function on the joke
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})


# -------------------------------
# 9. Define final pipeline
# -------------------------------
# Full flow:
#   Input → joke_gen_chain → parallel_chain
# Step 1: Generate joke
# Step 2: Run joke through both Passthrough + word_count in parallel
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)


# -------------------------------
# 10. Run the chain
# -------------------------------
# Example input: topic = "AI"
result = final_chain.invoke({'topic': 'AI'})

# result is a dict like:
# {'joke': 'Why did the AI cross the road?...', 'word_count': 12}

# -------------------------------
# 11. Format final result
# -------------------------------
# Combine joke and word count into a printable string
final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

# Print the joke and its word count
print(final_result)