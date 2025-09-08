# -------------------------------------------
# 1. Imports
# -------------------------------------------
from langgraph.graph import StateGraph, START, END   # LangGraph workflow builder
from langchain_openai import ChatOpenAI             # OpenAI wrapper for LangChain
from typing import TypedDict, Annotated             # Typed state definitions
from dotenv import load_dotenv                      # Load .env variables (API keys)
from pydantic import BaseModel, Field               # Define structured outputs
import operator                                     # For aggregation logic

# Load environment variables (ensures API keys are available)
load_dotenv()


# -------------------------------------------
# 2. Initialize LLM model
# -------------------------------------------
# Create an OpenAI model wrapper (gpt-4o-mini in this case)
# Setting temperature=0 ensures deterministic / stable responses
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# -------------------------------------------
# 3. Define schema for structured output
# -------------------------------------------
# Each evaluation (language/analysis/clarity) should return feedback + score.
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=1, le=10)

# Wrap the model so it always produces output matching EvaluationSchema
structured_model = model.with_structured_output(EvaluationSchema)


# -------------------------------------------
# 4. Define workflow state
# -------------------------------------------
class LLMState(TypedDict):
    essay: str                         # Input essay
    language_feedback: str             # Feedback about language quality
    analysis_feedback: str             # Feedback about depth of analysis
    clarity_feedback: str              # Feedback about clarity of thought
    overall_feedback: str              # Combined final feedback
    individual_scores: Annotated[list[int], operator.add]
    # `individual_scores` collects scores from multiple nodes in parallel.
    # The `Annotated[list[int], operator.add]` tells LangGraph how to merge updates:
    # instead of overwriting, it concatenates lists using `+`. For example:
    #   Language node returns {"individual_scores": [7]}
    #   Analysis node returns {"individual_scores": [8]}
    #   Clarity node returns {"individual_scores": [6]}
    # LangGraph merges them → [7] + [8] + [6] = [7, 8, 6]# Collect all scores
    average_score: float               # Final average score


# -------------------------------------------
# 5. Node functions (evaluations)
# -------------------------------------------

# (a) Evaluate language quality
def evaluate_language(state: LLMState):
    prompt = (
        f"Evaluate the language quality of the following essay and "
        f"provide feedback and a score out of 10:\n{state['essay']}"
    )
    output = structured_model.invoke(prompt)   # Returns EvaluationSchema
    return {
        "language_feedback": output.feedback,  # Save feedback string
        "individual_scores": [output.score]    # Append score to list
    }


# (b) Evaluate depth of analysis
def evaluate_analysis(state: LLMState):
    prompt = (
        f"Evaluate the depth of analysis of the following essay and "
        f"provide feedback and a score out of 10:\n{state['essay']}"
    )
    output = structured_model.invoke(prompt)
    return {
        "analysis_feedback": output.feedback,
        "individual_scores": [output.score]
    }


# (c) Evaluate clarity of thought
def evaluate_thought(state: LLMState):
    prompt = (
        f"Evaluate the clarity of thought of the following essay and "
        f"provide feedback and a score out of 10:\n{state['essay']}"
    )
    output = structured_model.invoke(prompt)
    return {
        "clarity_feedback": output.feedback,
        "individual_scores": [output.score]
    }


# (d) Final evaluation — summarize feedback + compute average score
def final_evaluation(state: LLMState):
    # Combine previous feedbacks into a single prompt
    prompt = (
        f"Based on the following feedbacks, create a summarized overall feedback:\n"
        f"Language feedback - {state['language_feedback']}\n"
        f"Depth of analysis feedback - {state['analysis_feedback']}\n"
        f"Clarity of thought feedback - {state['clarity_feedback']}"
    )

    # Use normal model (not structured) for free-text summary
    overall_feedback = model.invoke(prompt).content

    # Compute average score across evaluations
    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])

    return {"overall_feedback": overall_feedback, "average_score": avg_score}


# -------------------------------------------
# 6. Build workflow graph
# -------------------------------------------
graph = StateGraph(LLMState)

# Add nodes (processing steps)
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# Define execution flow:
# Start → run all 3 evaluations (parallelizable) → final evaluation → End
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

# Print results
print("Final Feedback:", result["overall_feedback"])
print("Average Score:", result["average_score"])