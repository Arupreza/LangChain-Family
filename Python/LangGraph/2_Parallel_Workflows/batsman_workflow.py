# -------------------------------------------
# 1. Import required libraries
# -------------------------------------------
from langgraph.graph import StateGraph, START, END  # LangGraph workflow builder
from typing import TypedDict  # For structured state definitions

# -------------------------------------------
# 2. Define the state schema for a batsman
# -------------------------------------------
class BatsmanState(TypedDict):
    runs: int                 # Total runs scored
    balls: int                # Total balls faced
    fours: int                # Number of 4s hit
    sixes: int                # Number of 6s hit
    sr: float                 # Strike rate (runs/balls * 100)
    bpb: float                # Balls per boundary (balls / (fours+sixes))
    boundary_percent: float   # % of runs that came from boundaries


# -------------------------------------------
# 3. Define calculation functions (nodes)
# -------------------------------------------
def calculate_sr(state: BatsmanState):
    """Calculate Strike Rate (SR)"""
    sr = (state['runs'] / state['balls'] * 100) / 100
    state['sr'] = sr
    return {'sr': sr}


def calculate_bpb(state: BatsmanState):
    """Calculate Balls per Boundary (BPB)"""
    bpb = state['balls'] / (state['fours'] + state['sixes'])
    return {'bpb': bpb}


def calculate_boundary_percent(state: BatsmanState):
    """Calculate % of runs from boundaries"""
    boundary_percent = (((state['fours'] * 4) + (state['sixes'] * 6)) / state['runs']) * 100
    return {'boundary_percent': boundary_percent}


def summary(state: BatsmanState):
    """Summarize all calculated metrics"""
    summary = f"""
    Strike Rate        - {state['sr']} \n
    Balls per boundary - {state['bpb']} \n
    Boundary percent   - {state['boundary_percent']}
    """
    return {'summary': summary}


# -------------------------------------------
# 4. Build the workflow graph
# -------------------------------------------
graph = StateGraph(BatsmanState)

# Add computation nodes
graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary_percent', calculate_boundary_percent)
graph.add_node('summary', summary)

# Define edges (execution flow)
graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_percent')

# After metrics → go to summary
graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary_percent', 'summary')

# End workflow after summary
graph.add_edge('summary', END)

# Compile the workflow
workflow = graph.compile()

# -------------------------------------------
# 5. Input initial state (match statistics)
# -------------------------------------------
initial_state = {
    'runs': 100,   # Example: Player scored 100 runs
    'balls': 50,   # Faced 50 balls
    'fours': 6,    # Hit 6 fours
    'sixes': 4     # Hit 4 sixes
}

# -------------------------------------------
# 6. Run workflow and print results
# -------------------------------------------
workflow_out = workflow.invoke(initial_state)
print(workflow_out)