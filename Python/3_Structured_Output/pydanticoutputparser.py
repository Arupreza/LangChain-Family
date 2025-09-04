# -------------------------------
# 1. Import required libraries
# -------------------------------
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables from .env (e.g., HuggingFace API keys)
load_dotenv()


# -------------------------------
# 2. Define model identifiers
# -------------------------------
# These are Hugging Face Hub model IDs.
# Here we will actually use the Gemma model.
gemma_model = "google/gemma-2-2b-it"
llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# -------------------------------
# 3. Initialize Hugging Face pipeline
# -------------------------------
# This creates a text-generation pipeline (via transformers)
# with specific decoding parameters.
llm = HuggingFacePipeline.from_model_id(
    model_id=gemma_model,     # Which HF model to load
    task="text-generation",   # Use text generation task
    pipeline_kwargs=dict(
        temperature=0.5,      # Lower temp = more deterministic responses
        max_new_tokens=100,   # Limit the maximum output length
        return_full_text=False,  # Return only generated part, not the input
    )
)


# -------------------------------
# 4. Wrap pipeline into LangChain Chat model
# -------------------------------
# This makes the HF pipeline compatible with LangChain's "Chat" interface.
model = ChatHuggingFace(llm=llm, model_id=gemma_model)


# -------------------------------
# 5. Define structured output schema
# -------------------------------
# Using Pydantic, we define the format of the expected result.
# This ensures output will be validated & parsed into a Python object.
class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')  # must be > 18
    city: str = Field(description='Name of the city the person belongs to')


# -------------------------------
# 6. Create Output Parser
# -------------------------------
# The parser will enforce that model outputs follow the Person schema.
parser = PydanticOutputParser(pydantic_object=Person)


# -------------------------------
# 7. Define Prompt Template
# -------------------------------
# The template asks the model to generate structured information
# about a fictional person from a given place.
template = PromptTemplate(
    template=(
        "Generate the name, age and city of a fictional {place} person\n"
        "{format_instruction}"   # parser's formatting instructions are inserted here
    ),
    input_variables=['place'],   # Placeholder that will be filled at runtime
    partial_variables={'format_instruction': parser.get_format_instructions()}
)


# -------------------------------
# 8. Build the Chain (Prompt → Model → Parser)
# -------------------------------
# Using LangChain Expression Language (LCEL), we "pipe" components together.
# 1) PromptTemplate → formats the input
# 2) Model → generates text
# 3) Parser → validates & parses output into Person object
chain = template | model | parser


# -------------------------------
# 9. Run the Chain with Input
# -------------------------------
# Input: place = "Bangladesh"
# Steps:
#   - Prompt is filled: "Generate the name, age and city of a fictional Bangladesh person ..."
#   - Model generates structured text
#   - Parser turns text into a Person object
final_result = chain.invoke({'place': 'Bangladesh'})

# Print the validated Person object
print(final_result)
# Example output: Person(name='Arjun Perera', age=28, city='Colombo')
