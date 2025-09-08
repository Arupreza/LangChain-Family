from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFacePipeline
from typing import TypedDict
from dotenv import load_dotenv
import re, torch, gc

# ---------------- GPU cleanup (optional) ----------------
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# ---------------- Load env ----------------
load_dotenv()

# ---------------- Model ----------------
gemma_model = "google/gemma-2-2b-it"

model = HuggingFacePipeline.from_model_id(
    model_id=gemma_model,
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.3,       # a bit more deterministic for scoring
        max_new_tokens=180,
        return_full_text=False
    ),
)

# ---------------- State schema ----------------
class BlogState(TypedDict):
    title: str
    outline: str
    content: str
    content_score: int  # 1..10

# ---------------- Nodes ----------------
def create_outline(state: BlogState) -> BlogState:
    title = state["title"]
    prompt = (
        "Create a concise, hierarchical outline (H1/H2/H3) for a blog titled:\n"
        f"'{title}'. Keep it structured."
    )
    outline = model.invoke(prompt)
    state["outline"] = outline if isinstance(outline, str) else str(outline)
    return state

def create_blog(state: BlogState) -> BlogState:
    title = state["title"]
    outline = state["outline"]
    prompt = (
        "Write a clear, well-structured blog post using the outline.\n"
        f"Title: {title}\n\nOutline:\n{outline}\n\n"
        "Use headings, short paragraphs, and end with 3 actionable takeaways."
    )
    content = model.invoke(prompt)
    state["content"] = content if isinstance(content, str) else str(content)
    return state

def score_content(state: BlogState) -> BlogState:
    """Ask the model for a 1..10 score and parse it robustly."""
    content = state["content"]
    prompt = (
        "Rate the quality of the following blog content on a scale of 1 to 10.\n"
        "Respond with ONLY the integer (no words, no punctuation).\n\n"
        f"{content}\n\n"
        "Score:"
    )
    score_text = model.invoke(prompt)
    score_text = score_text if isinstance(score_text, str) else str(score_text)

    # Extract first integer 1..10
    m = re.search(r"\b(10|[1-9])\b", score_text)
    score = int(m.group(1)) if m else 0   # default to 0 if parsing fails
    # Clamp just in case
    score = max(0, min(10, score))
    state["content_score"] = score
    return state

# ---------------- Graph ----------------
graph = StateGraph(BlogState)
graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)
graph.add_node("score_content", score_content)

# Correct flow: END after scoring (remove the early END edge)
graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", "score_content")
graph.add_edge("score_content", END)

workflow = graph.compile()

# ---------------- Run ----------------
initial_state: BlogState = {"title": "The Future of AI in Healthcare", "outline": "", "content": "", "content_score": 0}
final_state = workflow.invoke(initial_state)

print("\n=== OUTLINE ===\n", final_state["outline"])
print("\n=== CONTENT ===\n", final_state["content"])
print("\n=== SCORE ===\n", final_state["content_score"])
