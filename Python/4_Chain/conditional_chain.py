# -------------------------------
# 1. Import required libraries
# -------------------------------
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

# Load environment variables (Hugging Face keys, etc.)
load_dotenv()

# -------------------------------
# 2. Define output parsers
# -------------------------------
# StrOutputParser → just gives plain text
parser_1 = StrOutputParser()

# -------------------------------
# 3. Define model IDs
# -------------------------------
gemma_model = "google/gemma-2-2b-it"
llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# -------------------------------
# 4. Initialize Hugging Face pipeline
# -------------------------------
# Using Gemma 2B IT as the base model for generation
model = HuggingFacePipeline.from_model_id(
    model_id=gemma_model,
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,
        return_full_text=False
    )
)

# -------------------------------
# 5. Define structured output model
# -------------------------------
# Sentiment classification schema: only 3 allowed values
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(
        description='Give the sentiment of the feedback'
    )

# Parser that validates LLM output against the schema
parser_2 = PydanticOutputParser(pydantic_object=Feedback)

# -------------------------------
# 6. Define classification prompt
# -------------------------------
# This asks the LLM to classify feedback into positive/negative/neutral
# and return a strict JSON object matching the schema.
prompt_1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback text "
        "into one of: positive, negative, or neutral.\n\n"
        "Feedback: {feedback}\n\n"
        "Return ONLY a valid JSON object in this format:\n{format_instruction}"
    ),
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser_2.get_format_instructions()}
)

# Build classifier chain: (prompt → model → parser)
classifier_chain = prompt_1 | model | parser_2


# -------------------------------
# 7. Define branch prompts (responses)
# -------------------------------
# If sentiment is POSITIVE → thank the customer positively
prompt_2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

# If sentiment is NEGATIVE → apologize and respond accordingly
prompt_3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


# -------------------------------
# 8. Define branching logic
# -------------------------------
# RunnableBranch lets us conditionally select the next chain
branch_chain = RunnableBranch(
    # Case 1: If classified sentiment == "positive"
    (lambda x: x.sentiment == 'positive', prompt_2 | model | parser_1),

    # Case 2: If classified sentiment == "negative"
    (lambda x: x.sentiment == 'negative', prompt_3 | model | parser_1),

    # Default: If "neutral" or anything else → generic thank-you
    RunnableLambda(lambda x: "Thank you for your feedback.")
)


# -------------------------------
# 9. Build the full pipeline
# -------------------------------
# Flow:
#   classifier_chain → decides sentiment
#   branch_chain → generates response based on sentiment
chain = classifier_chain | branch_chain


# -------------------------------
# 10. Run the chain
# -------------------------------
# Input: positive feedback
# 1) classifier_chain → returns Feedback(sentiment='positive')
# 2) branch_chain → triggers positive response chain
print(chain.invoke({'feedback': 'The product quality is excellent and delivery was prompt!'}))


# -------------------------------
# 11. Visualize chain as ASCII graph
# -------------------------------
# This prints the execution flow as a tree diagram
chain.get_graph().print_ascii()
