# ======================================================
# Tweet Generation Workflow with LangGraph + OpenAI
# ======================================================
# This workflow:
# 1. Generates a tweet on a given topic.
# 2. Evaluates the tweet for humor, originality, and virality.
# 3. If approved → stop.
#    If not approved → optimize the tweet and re-evaluate.
# 4. Iterates until either an "approved" tweet is found
#    or max iterations is reached.
#
# Key LLM Roles:
# - generator_llm: Writes funny tweets.
# - evaluator_llm: Harsh critic that checks quality.
# - optimizer_llm: Improves rejected tweets.
# ======================================================

from langgraph.graph import StateGraph, START, END       # Build workflows
from typing import TypedDict, Literal, Annotated         # State typing + reducer
from langchain_openai import ChatOpenAI                  # OpenAI chat model
from dotenv import load_dotenv                           # Load API key
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field                    # Structured schema
import operator

# ------------------------------------------------------
# 1. Setup
# ------------------------------------------------------
# Load environment variables (API key required for ChatOpenAI)
load_dotenv()

# Define three different LLM roles
generator_llm = ChatOpenAI(model='gpt-4o-mini')
evaluator_llm = ChatOpenAI(model='gpt-4o-mini')
optimizer_llm = ChatOpenAI(model='gpt-4o-mini')

# ------------------------------------------------------
# 2. Define Schema for Evaluator Output
# ------------------------------------------------------
class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        ..., description="Final evaluation result."
    )
    feedback: str = Field(..., description="Feedback for the tweet.")

# Wrap evaluator LLM with structured output
structured_evaluator_llm = evaluator_llm.with_structured_output(TweetEvaluation)

# ------------------------------------------------------
# 3. Define State for Workflow
# ------------------------------------------------------
class TweetState(TypedDict):
    topic: str                                # Topic of the tweet
    tweet: str                                # Current tweet text
    evaluation: Literal["approved", "needs_improvement"]  # Latest evaluation
    feedback: str                             # Latest feedback
    iteration: int                            # Current optimization iteration
    max_iteration: int                        # Max iterations allowed

    # Histories → use Annotated with operator.add to concatenate results
    tweet_history: Annotated[list[str], operator.add]       # Track all tweets tried
    feedback_history: Annotated[list[str], operator.add]    # Track all feedbacks

# ------------------------------------------------------
# 4. Node Functions
# ------------------------------------------------------

# --- Generate a tweet ---
def generate_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
        Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

        Rules:
        - Do NOT use question-answer format.
        - Max 280 characters.
        - Use observational humor, irony, sarcasm, or cultural references.
        - Think in meme logic, punchlines, or relatable takes.
        - Use simple, day to day english
        """)
    ]

    response = generator_llm.invoke(messages).content
    return {'tweet': response, 'tweet_history': [response]}


# --- Evaluate a tweet ---
def evaluate_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. "
                              "You evaluate tweets based on humor, originality, virality, and format."),
        HumanMessage(content=f"""
        Evaluate the following tweet:

        Tweet: "{state['tweet']}"

        Criteria:
        1. Originality – Fresh or overused?
        2. Humor – Does it make you smile/laugh?
        3. Punchiness – Short, sharp, scroll-stopping?
        4. Virality – Likely to be retweeted/shared?
        5. Format – Proper tweet (not Q&A, not setup-punchline, <280 chars)?

        Auto-reject if:
        - Written in question-answer format
        - Exceeds 280 characters
        - Reads like a setup-punchline joke
        - Ends with generic throwaway/vague lines

        Respond ONLY in structured format:
        - evaluation: "approved" or "needs_improvement"
        - feedback: paragraph with strengths & weaknesses
        """)
    ]

    response = structured_evaluator_llm.invoke(messages)  # Returns TweetEvaluation
    return {
        'evaluation': response.evaluation,
        'feedback': response.feedback,
        'feedback_history': [response.feedback]
    }


# --- Optimize a tweet based on feedback ---
def optimize_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
        Improve the tweet based on this feedback:
        "{state['feedback']}"

        Topic: "{state['topic']}"
        Original Tweet:
        {state['tweet']}

        Re-write it as a short, viral-worthy tweet.
        - Avoid Q&A style
        - Stay under 280 characters
        """)
    ]

    response = optimizer_llm.invoke(messages).content
    iteration = state['iteration'] + 1  # Increase iteration count

    return {'tweet': response, 'iteration': iteration, 'tweet_history': [response]}


# --- Router: Decide what to do after evaluation ---
def route_evaluation(state: TweetState):
    # Stop if approved OR max iterations reached
    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvement'

# ------------------------------------------------------
# 5. Build Workflow Graph
# ------------------------------------------------------
graph = StateGraph(TweetState)

# Add nodes
graph.add_node('generate', generate_tweet)
graph.add_node('evaluate', evaluate_tweet)
graph.add_node('optimize', optimize_tweet)

# Define flow
graph.add_edge(START, 'generate')   # Start → Generate
graph.add_edge('generate', 'evaluate')  # Evaluate generated tweet

# Conditional edge: evaluation result decides path
graph.add_conditional_edges(
    'evaluate', 
    route_evaluation,
    {
        'approved': END,               # If approved → finish
        'needs_improvement': 'optimize'  # If not → optimize & retry
    }
)

# After optimization → re-evaluate
graph.add_edge('optimize', 'evaluate')

# Compile workflow
workflow = graph.compile()

# ------------------------------------------------------
# 6. Test Workflow
# ------------------------------------------------------
initial_state = {
    "topic": "Bangladesh Politics",
    "iteration": 1,     # Start from 1st iteration
    "max_iteration": 5  # Allow up to 5 retries
}

result = workflow.invoke(initial_state)

# Print final result (state contains tweet, feedback, history, etc.)
print("=== Final Tweet ===")
print(result['tweet'])
print("\n=== Evaluation ===")
print(result['evaluation'])
print("\n=== Feedback History ===")
for fb in result['feedback_history']:
    print("-", fb)