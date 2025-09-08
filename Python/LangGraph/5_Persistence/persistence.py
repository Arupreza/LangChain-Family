# ======================================================
# Essay Evaluation Workflow with LangGraph + OpenAI + FAISS
# ======================================================
# Workflow steps:
# 1. Takes an essay as input.
# 2. Evaluates it across 3 dimensions:
#    - Language quality
#    - Depth of analysis
#    - Clarity of thought
# 3. Produces a final summary + average score.
# 4. Stores essay + evaluation in a FAISS vector database
#    so that all results can be searched semantically later.
#
# Persistence:
# - At session start → Load FAISS history (if exists).
# - If no index exists → create new FAISS index with InMemoryDocstore.
# - Save new evaluations to disk in the same folder as this script.
# ======================================================

# -------------------------------------------
# 1. Imports
# -------------------------------------------
import os
import operator
import faiss
import datetime
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END   # LangGraph workflow builder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

# -------------------------------------------
# 2. Load environment variables
# -------------------------------------------
# Make sure you have OPENAI_API_KEY in your .env file
load_dotenv()

# -------------------------------------------
# 3. Initialize LLM model + Embeddings
# -------------------------------------------
# gpt-4o-mini → fast and cheap LLM
# temperature=0 → deterministic outputs
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# OpenAI Embedding model → 1536-dim vectors
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -------------------------------------------
# 4. Initialize or Load FAISS index (persistence)
# -------------------------------------------
# Ensure FAISS DB is saved in the same folder as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
faiss_path = os.path.join(script_dir, "essay_faiss_index")

if os.path.exists(faiss_path):
    # Load existing FAISS index from disk
    index = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    print("📂 Loaded existing FAISS history.")
else:
    # Create an empty FAISS index
    dimension = 1536  # must match embedding size
    faiss_index = faiss.IndexFlatL2(dimension)
    index = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    print("🆕 Created new FAISS index.")

# -------------------------------------------
# 5. Define schema for structured output
# -------------------------------------------
# Ensures LLM returns JSON with feedback + score
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=1, le=10)

# Wrap model with schema enforcement
structured_model = model.with_structured_output(EvaluationSchema)

# -------------------------------------------
# 6. Define Workflow State
# -------------------------------------------
# This is like memory shared across nodes in LangGraph
class LLMState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]  # merge scores
    average_score: float

# -------------------------------------------
# 7. Node functions (LangGraph nodes)
# -------------------------------------------
def evaluate_language(state: LLMState):
    """Evaluate language quality of essay"""
    prompt = f"Evaluate the language quality:\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {"language_feedback": output.feedback, "individual_scores": [output.score]}

def evaluate_analysis(state: LLMState):
    """Evaluate depth of analysis of essay"""
    prompt = f"Evaluate the depth of analysis:\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {"analysis_feedback": output.feedback, "individual_scores": [output.score]}

def evaluate_thought(state: LLMState):
    """Evaluate clarity of thought of essay"""
    prompt = f"Evaluate the clarity of thought:\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {"clarity_feedback": output.feedback, "individual_scores": [output.score]}

def final_evaluation(state: LLMState):
    """Summarize all feedback + compute average score"""
    prompt = (
        f"Summarize the essay feedbacks:\n"
        f"- Language: {state['language_feedback']}\n"
        f"- Analysis: {state['analysis_feedback']}\n"
        f"- Clarity: {state['clarity_feedback']}"
    )
    overall_feedback = model.invoke(prompt).content
    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])
    return {"overall_feedback": overall_feedback, "average_score": avg_score}

# -------------------------------------------
# 8. Build LangGraph workflow
# -------------------------------------------
graph = StateGraph(LLMState)
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# Execution flow: Start → parallel evals → final → End
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()

# -------------------------------------------
# 9. Run Workflow on a Test Essay
# -------------------------------------------
essay2 = """India and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI).
India also want become big in this AI thing. If work hard, India can go top.
But if no careful, India go back.

India have many good. We have smart student, many engine-ear, and good IT peoples.
Big company like TCS, Infosys, Wipro already use AI. Government also do program “AI for All”.
It want AI in farm, doctor place, school and transport.
"""

initial_state = {"essay": essay2}
result = workflow.invoke(initial_state)

print("\n=== Final Feedback ===")
print(result["overall_feedback"])
print("\n=== Average Score ===")
print(result["average_score"])

# -------------------------------------------
# 10. Save Results into FAISS
# -------------------------------------------
# Combine essay + feedbacks into one text chunk
conversation_text = f"""
Essay: {result['essay']}

Language Feedback: {result['language_feedback']}
Analysis Feedback: {result['analysis_feedback']}
Clarity Feedback: {result['clarity_feedback']}

Overall Feedback: {result['overall_feedback']}
Average Score: {result['average_score']}
"""

# Metadata for filtering (score, type, timestamp)
metadata = {
    "average_score": result["average_score"],
    "type": "essay_evaluation",
    "timestamp": datetime.datetime.now().isoformat()
}

# Add new document into FAISS
index.add_texts([conversation_text], metadatas=[metadata])

# Save FAISS DB locally (in same folder as script)
index.save_local(faiss_path)
print("\n✅ New conversation saved in FAISS history.")

# -------------------------------------------
# 11. Example: Semantic Search over FAISS
# -------------------------------------------
query = "AI in education"
similar = index.similarity_search(query, k=2)

print("\n=== Semantic Search Results ===")
for doc in similar:
    print(doc.page_content[:200], "...")  # preview
    print("Metadata:", doc.metadata)
    print("------")