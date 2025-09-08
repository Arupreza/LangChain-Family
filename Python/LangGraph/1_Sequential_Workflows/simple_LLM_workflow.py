# ✅ Imports
from langgraph.graph import StateGraph, START, END   # StateGraph = workflow builder
from langchain_huggingface import HuggingFacePipeline  # wrapper to use HF models in LangChain
from typing import TypedDict   # define workflow state schema
from dotenv import load_dotenv # load env variables (like Hugging Face token)

# ---------------------------------------------
# 1. Load environment variables
# ---------------------------------------------
# If your HuggingFace model requires an API token (gated/private models),
# store it in a `.env` file as HUGGINGFACEHUB_API_TOKEN
load_dotenv()

# ---------------------------------------------
# 2. Choose a Hugging Face model
# ---------------------------------------------
gemma_model = "google/gemma-2-2b-it"  # small, instruction-tuned model
# llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # alternative demo model

# ---------------------------------------------
# 3. Wrap the HF model as a LangChain LLM
# ---------------------------------------------
# HuggingFacePipeline.from_model_id loads the model + tokenizer
# and returns a LangChain-compatible LLM object.
model = HuggingFacePipeline.from_model_id(
    model_id=gemma_model,
    task="text-generation",   # type of pipeline we want
    pipeline_kwargs=dict(
        temperature=0.5,      # controls randomness (lower = deterministic)
        max_new_tokens=100,   # maximum tokens to generate per response
        return_full_text=False # return only the answer (not the prompt+answer)
    ),
)

# ---------------------------------------------
# 4. Define the workflow state schema
# ---------------------------------------------
# The workflow passes around a dictionary that must follow this schema.
class LLMState(TypedDict):
    user_input: str      # input question
    model_response: str  # output answer (LLM result)

# ---------------------------------------------
# 5. Define a node function
# ---------------------------------------------
# Each node takes a state dict -> updates it -> returns it.
def llm_qa(state: LLMState) -> LLMState:
    # ✅ Extract the user question from the state
    question = state["user_input"]

    # ✅ Create a simple prompt
    prompt = (
        "Answer the question based on the context below.\n\n"
        f"Context: {question}\n\n"
        "Answer:"
    )

    # ✅ Call the Hugging Face model (works like a LangChain LLM)
    result = model.invoke(prompt)

    # ✅ Normalize output: usually it's already a string
    state["model_response"] = result.strip() if isinstance(result, str) else str(result).strip()

    # ✅ Return updated state
    return state

# ---------------------------------------------
# 6. Build the LangGraph workflow
# ---------------------------------------------
graph = StateGraph(LLMState)          # create graph with our state schema
graph.add_node("llm_qa", llm_qa)      # add one node (our QA function)
graph.add_edge(START, "llm_qa")       # connect START → llm_qa
graph.add_edge("llm_qa", END)         # connect llm_qa → END

# ---------------------------------------------
# 7. Compile the graph into an executable workflow
# ---------------------------------------------
workflow = graph.compile()

# ---------------------------------------------
# 8. Run the workflow with some input
# ---------------------------------------------
# Keys must match our LLMState schema: use 'user_input', not 'question'
initial_state = {"user_input": "Tell me about Bangladesh."}

# Run the compiled workflow
final_state = workflow.invoke(initial_state)

# Print the result
print(final_state['model_response'])
# Example output:
# {
#   'user_input': 'Tell me about Bangladesh.',
#   'model_response': 'Bangladesh is a country in South Asia...'
# }
