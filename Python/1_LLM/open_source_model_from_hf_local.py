# . check the mode available in the local machine 
# ls ~/.cache/huggingface/hub

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

gemma_model = "google/gemma-2-2b-it"
llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

llm = HuggingFacePipeline.from_model_id(
    model_id=gemma_model,
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,
        return_full_text=False,
    )
)

model = ChatHuggingFace(llm=llm, model_id=gemma_model)

result = model.invoke("What is the capital of Bangladesh?")
print(result.content)


