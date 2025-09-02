# . check the mode available in the local machine 
# ls ~/.cache/huggingface/hub

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Build the Hugging Face pipeline wrapper
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,   # <-- correct key
    )
)

# Wrap in a chat interface
model = ChatHuggingFace(llm=llm)

# Run inference
result = model.invoke("What is the capital of Bangladesh?")
print(result.content)   # .content gives just the text output
