# ======================================================
# Essay Evaluation Workflow with LangGraph + OpenAI
# ======================================================
# This workflow:
# 1. Takes an essay as input.
# 2. Evaluates it across three dimensions:
#    - Language quality
#    - Depth of analysis
#    - Clarity of thought
# 3. Collects all scores and feedback in parallel.
# 4. Produces a final summary feedback + average score.
#
# Key Concepts:
# - StateGraph → models the workflow (nodes + edges).
# - Structured output → ensures clean JSON-style feedback & score.
# - Annotated[list[int], operator.add] → merges scores from multiple nodes.
# ======================================================

# -------------------------------------------
# 1. Imports
# -------------------------------------------
from langgraph.graph import StateGraph, START, END    # LangGraph workflow builder
from langchain_openai import ChatOpenAI              # OpenAI wrapper for LangChain
from typing import TypedDict, Annotated              # TypedDict for state, Annotated for merging lists
from dotenv import load_dotenv                       # Loads .env (API keys, etc.)
from pydantic import BaseModel, Field                # Schema definitions
import operator                                      # Operator used for merging values

# Load environment variables (ensures OPENAI_API_KEY is available)
load_dotenv()


# -------------------------------------------
# 2. Initialize LLM model
# -------------------------------------------
# gpt-4o-mini chosen for speed and cost efficiency.
# temperature=0 makes it deterministic (stable outputs).
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# -------------------------------------------
# 3. Define schema for structured output
# -------------------------------------------
# Each evaluation returns:
# - feedback (text explanation)
# - score (int 1–10)
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=1, le=10)

# Wrap LLM → ensures evaluation responses always match schema
structured_model = model.with_structured_output(EvaluationSchema)


# -------------------------------------------
# 4. Define workflow state
# -------------------------------------------
# This is the "memory" carried through the workflow.
class LLMState(TypedDict):
    essay: str                         # Input essay text
    language_feedback: str             # Feedback on language
    analysis_feedback: str             # Feedback on analysis
    clarity_feedback: str              # Feedback on clarity
    overall_feedback: str              # Final combined summary feedback

    # `individual_scores` collects scores from multiple nodes in parallel.
    # The Annotated[...] + operator.add tells LangGraph to MERGE results
    # instead of overwriting. Example:
    #   Language → {"individual_scores": [7]}
    #   Analysis → {"individual_scores": [8]}
    #   Clarity  → {"individual_scores": [6]}
    # Merged → [7, 8, 6]
    individual_scores: Annotated[list[int], operator.add]

    average_score: float               # Final averaged score


# -------------------------------------------
# 5. Node functions (evaluations)
# -------------------------------------------

# (a) Language quality
def evaluate_language(state: LLMState):
    prompt = f"Evaluate the language quality of this essay and provide feedback and a score (1–10):\n{state['essay']}"
    output = structured_model.invoke(prompt)   # Returns EvaluationSchema
    return {
        "language_feedback": output.feedback,
        "individual_scores": [output.score]
    }

# (b) Depth of analysis
def evaluate_analysis(state: LLMState):
    prompt = f"Evaluate the depth of analysis of this essay and provide feedback and a score (1–10):\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {
        "analysis_feedback": output.feedback,
        "individual_scores": [output.score]
    }

# (c) Clarity of thought
def evaluate_thought(state: LLMState):
    prompt = f"Evaluate the clarity of thought in this essay and provide feedback and a score (1–10):\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {
        "clarity_feedback": output.feedback,
        "individual_scores": [output.score]
    }

# (d) Final evaluation → Summarize feedbacks + compute average
def final_evaluation(state: LLMState):
    # Summarize feedbacks into a single paragraph
    prompt = (
        f"Create a summary of the essay feedbacks:\n"
        f"- Language feedback: {state['language_feedback']}\n"
        f"- Analysis feedback: {state['analysis_feedback']}\n"
        f"- Clarity feedback: {state['clarity_feedback']}"
    )
    overall_feedback = model.invoke(prompt).content

    # Compute average score
    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])

    return {"overall_feedback": overall_feedback, "average_score": avg_score}


# -------------------------------------------
# 6. Build workflow graph
# -------------------------------------------
graph = StateGraph(LLMState)

# Add nodes
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# Define execution flow:
# Start → three parallel evaluations → final evaluation → End
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

# Compile the graph into a runnable workflow
workflow = graph.compile()


# -------------------------------------------
# 7. Test Essay Input
# -------------------------------------------
essay2 = """India and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI). 
India also want become big in this AI thing. If work hard, India can go top. 
But if no careful, India go back.

India have many good. We have smart student, many engine-ear, and good IT peoples. 
Big company like TCS, Infosys, Wipro already use AI. Government also do program “AI for All”. 
It want AI in farm, doctor place, school and transport.

In farm, AI help farmer know when to put seed, when rain come, how stop bug. 
In health, AI help doctor see sick early. In school, AI help student learn good. 
Government office use AI to find bad people and work fast.

But problem come also. First is many villager no have phone or internet. So AI not help them. 
Second, many people lose job because AI and machine do work. Poor people get more bad.

One more big problem is privacy. AI need big big data. Who take care? 
India still make data rule. If no strong rule, AI do bad.

India must all people together – govern, school, company and normal people. 
We teach AI and make sure AI not bad. Also talk to other country and learn from them.

If India use AI good way, we become strong, help poor and make better life. 
But if only rich use AI, and poor no get, then big bad thing happen.

So, in short, AI time in India have many hope and many danger. 
We must go right road. AI must help all people, not only some. 
Then India grow big and world say "good job India".
"""

# Initialize workflow state with essay text
initial_state = {"essay": essay2}

# Run workflow
result = workflow.invoke(initial_state)

# -------------------------------------------
# 8. Print Results
# -------------------------------------------
print("=== Final Feedback ===")
print(result["overall_feedback"])
print("\n=== Average Score ===")
print(result["average_score"])