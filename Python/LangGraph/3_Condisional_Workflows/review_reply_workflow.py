# ======================================================
# Review Reply Workflow with LangGraph + OpenAI
# ======================================================
# This workflow:
# 1. Takes a user review as input.
# 2. Classifies sentiment (positive, negative, neutral).
# 3. Depending on sentiment:
#       - Positive → Thank-you response.
#       - Neutral  → Polite professional response.
#       - Negative → Run a diagnosis (issue type, tone, urgency),
#                    then craft an empathetic resolution response.
# ======================================================

from langgraph.graph import StateGraph, START, END       # For building workflow graph
from langchain_openai import ChatOpenAI                  # OpenAI LLM wrapper
from typing import TypedDict, Literal                    # Structured state typing
from dotenv import load_dotenv                           # Load API key from .env
from pydantic import BaseModel, Field                    # Define structured outputs

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

# ------------------------------------------------------
# 1. Initialize OpenAI LLM model
# ------------------------------------------------------
# - Using gpt-4o-mini for speed + quality
# - temperature=0 ensures deterministic, repeatable outputs
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ------------------------------------------------------
# 2. Define Schemas for structured output
# ------------------------------------------------------
# Sentiment classification: one of [positive, negative, neutral]
class SentimentSchema(BaseModel):
    label: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment of the review"
    )

# For negative reviews: extract more details
class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"]
    tone: Literal["angry", "frustrated", "disappointed", "calm"]
    urgency: Literal["low", "medium", "high"]

# Wrap models with structured outputs
structured_model_1 = model.with_structured_output(SentimentSchema)
structured_model_2 = model.with_structured_output(DiagnosisSchema)

# ------------------------------------------------------
# 3. Define State of the Workflow
# ------------------------------------------------------
# This defines what information the graph will carry across nodes.
class ReviewState(TypedDict):
    review: str                                     # Input review text
    sentiment: Literal["positive", "negative", "neutral"]  # Classified sentiment
    diagnosis: dict                                 # For negative reviews: {issue_type, tone, urgency}
    response: str                                   # Final generated response to user

# ------------------------------------------------------
# 4. Node Functions
# ------------------------------------------------------

# Step 1: Find sentiment of review
def find_sentiment(state: ReviewState):
    prompt = f"Classify the sentiment of this review as positive, negative, or neutral:\n\n{state['review']}\n\nSentiment:"
    sentiment = structured_model_1.invoke(prompt)   # Returns SentimentSchema object
    return {'sentiment': sentiment.label}           # Extract label string

# Step 2: Router → Decide which branch to follow
def check_sentiment(state: ReviewState) -> str:
    """Directs flow to the appropriate node based on sentiment."""
    if state['sentiment'] == "positive":
        return "positive_response"   # → Thank-you
    elif state['sentiment'] == "negative":
        return "run_diagnosis"       # → Diagnose first, then handle negative
    else:
        return "neutral_response"    # → Polite neutral reply

# Step 3a: Generate thank-you for positive reviews
def positive_response(state: ReviewState):
    prompt = f"""Write a warm thank-you message in response to this review:
    \n\n\"{state['review']}\"\n Also, kindly ask the user to leave feedback 
    on our website."""
    response = model.invoke(prompt)
    return {'response': response.content}           # .content → plain text

# Step 3b (Negative branch): Run diagnosis
def run_diagnosis(state: ReviewState):
    prompt = f"""Diagnose this negative review:\n\n{state['review']}\n
    Return issue_type, tone, and urgency."""
    response = structured_model_2.invoke(prompt)   # Returns DiagnosisSchema
    return {'diagnosis': response.model_dump()}    # Convert to dict

# Step 3c: Generate empathetic resolution for negative reviews
def negative_response(state: ReviewState):
    diagnosis = state['diagnosis']   # Extract details
    prompt = f"""You are a support assistant.
    The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', 
    and marked urgency as '{diagnosis['urgency']}'.
    Write an empathetic, helpful resolution message."""
    response = model.invoke(prompt)
    return {'response': response.content}

# Step 3d: Neutral → polite response
def neutral_response(state: ReviewState):
    prompt = f"""Write a polite and professional response to this neutral review:
    \n\n\"{state['review']}\"\n Also, kindly ask the user to leave feedback 
    on our website."""
    response = model.invoke(prompt)
    return {'response': response.content}

# ------------------------------------------------------
# 5. Build the Workflow Graph
# ------------------------------------------------------
graph = StateGraph(ReviewState)

# Add nodes (steps)
graph.add_node('find_sentiment', find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('negative_response', negative_response)
graph.add_node('neutral_response', neutral_response)

# Add edges (flow rules)
graph.add_edge(START, 'find_sentiment')                   # Start → sentiment check
graph.add_conditional_edges('find_sentiment', check_sentiment)  # Branch logic
graph.add_edge('positive_response', END)                  # Positive → End
graph.add_edge('run_diagnosis', 'negative_response')      # Negative → Diagnosis → Response
graph.add_edge('negative_response', END)                  # Negative response → End
graph.add_edge('neutral_response', END)                   # Neutral → End

# Compile workflow into executable form
workflow = graph.compile()

# ------------------------------------------------------
# 6. Test the Workflow
# ------------------------------------------------------
initial_state = {
    'review': "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}

# Run the workflow with given review
result = workflow.invoke(initial_state)

# Print final auto-generated response
print("AI Response:\n", result['response'])