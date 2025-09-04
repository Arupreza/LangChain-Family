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
    RunnablePassthrough,
    RunnableBranch,
    RunnableLambda
)

# Load environment variables (e.g., Hugging Face token if required)
load_dotenv()


# -------------------------------
# 2. Define prompts
# -------------------------------
# Prompt 1: Generate a detailed report based on a topic
prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']   # takes a topic as input
)

# Prompt 2: Summarize text into a shorter version
prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']    # takes raw text as input
)


# -------------------------------
# 3. Define model IDs
# -------------------------------
gemma_model = "google/gemma-2-2b-it"
llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# -------------------------------
# 4. Initialize Hugging Face model pipeline
# -------------------------------
# Using Gemma as the main generator
model = HuggingFacePipeline.from_model_id(
    model_id=gemma_model,
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,       # controls randomness (lower = more deterministic)
        max_new_tokens=100,    # maximum new tokens to generate
        return_full_text=False # return only generated content, not the input
    )
)


# -------------------------------
# 5. Define output parser
# -------------------------------
# This converts the raw model output into plain text
parser = StrOutputParser()


# -------------------------------
# 6. Define report generation chain
# -------------------------------
# Sequence: (prompt1 → model → parser)
# Input: {"topic": "..."} → Output: full report text
report_gen_chain = prompt1 | model | parser


# -------------------------------
# 7. Define branching logic
# -------------------------------
# RunnableBranch chooses the next step depending on a condition:
#   - If report length > 100 words → summarize (prompt2 → model → parser)
#   - Otherwise → just return the report as-is (RunnablePassthrough)
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 100, prompt2 | model | parser),
    RunnablePassthrough()
)


# -------------------------------
# 8. Define final chain
# -------------------------------
# Full pipeline = first generate report, then apply branching
# Flow:
#   Input (topic) → report_gen_chain → branch_chain
final_chain = RunnableSequence(report_gen_chain, branch_chain)


# -------------------------------
# 9. Run the chain
# -------------------------------
# Example input: topic "Russia vs Ukraine"
#   1. Generate report (about 100 tokens)
#   2. If report is long (>100 words), summarize
#   3. Else, return the original report
print(final_chain.invoke({'topic': 'Russia vs Ukraine'}))