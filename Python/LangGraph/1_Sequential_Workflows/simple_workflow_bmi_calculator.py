# ✅ Import necessary modules
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Image


# 1. Define the state structure for our workflow
#    - TypedDict ensures the workflow state has a fixed schema:
#      weight (kg), height (m), and bmi (calculated value).
class BMIState(TypedDict):
    weight: float   # input: weight in kilograms
    height: float   # input: height in meters
    bmi: float      # output: body mass index


# 2. Define the function (a node in the graph) that calculates BMI
def calculate_bmi(state: BMIState) -> BMIState:
    # Extract inputs from the state dictionary
    weight = state["weight"]
    height = state["height"]

    # BMI formula: weight (kg) / (height in meters)^2
    state["bmi"] = round(weight / (height ** 2), 2)

    # Return the updated state dictionary (mandatory in LangGraph nodes)
    return state


# 3. Build the workflow graph
#    - StateGraph defines nodes (functions) and edges (execution flow).
graph = StateGraph(BMIState)

# Add a node named "calculate_bmi" mapped to our function
graph.add_node("calculate_bmi", calculate_bmi)

# Define execution order with edges:
# START → calculate_bmi → END
graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", END)


# 4. Compile the graph into an executable workflow
workflow = graph.compile()

# 5. Run the workflow with an initial input state
final_state = workflow.invoke({"weight": 70, "height": 1.75})

# Print the result (should include the calculated BMI)
print(final_state)  
# Example Output: {'weight': 70, 'height': 1.75, 'bmi': 22.86}


# 6. Visualize the workflow graph
#    - get_graph() produces a graph object
#    - draw_mermaid_png() renders a Mermaid diagram as PNG
display(Image(data=workflow.get_graph().draw_mermaid_png()))

# (Alternative)
# You can also visualize directly from the uncompiled graph:
# display(Image(data=graph.get_graph().draw_mermaid_png()))