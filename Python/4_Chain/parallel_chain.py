# -------------------------------
# 1. Import required libraries
# -------------------------------
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

# -------------------------------
# 2. Input text (example: SVM description)
# -------------------------------
# This will be used as input for generating notes and questions.
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

# Load environment variables (e.g., HuggingFace API keys)
load_dotenv()


# -------------------------------
# 3. Define model IDs
# -------------------------------
# We use two different Hugging Face models:
# - gemma_model (google Gemma 2B IT)
# - llama_model (TinyLlama 1.1B Chat)
gemma_model = "google/gemma-2-2b-it"
llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# -------------------------------
# 4. Initialize Hugging Face pipelines
# -------------------------------
# Create text-generation pipelines with parameters.
# NOTE: temperature only works if do_sample=True, otherwise it's ignored.
model_1 = HuggingFacePipeline.from_model_id(
    model_id=gemma_model, 
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5, 
        max_new_tokens=100, 
        return_full_text=False
    )
)

model_2 = HuggingFacePipeline.from_model_id(
    model_id=llama_model,
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5, 
        max_new_tokens=100, 
        return_full_text=False
    )
)


# -------------------------------
# 5. Define PromptTemplates
# -------------------------------
# Prompt 1: generate simplified notes from the input text
prompt_1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

# Prompt 2: generate 5 Q&A pairs from the input text
prompt_2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

# Prompt 3: merge the outputs (notes + quiz) into one document
prompt_3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)


# -------------------------------
# 6. Define output parser
# -------------------------------
# We only need plain strings, so we use StrOutputParser.
parser = StrOutputParser()


# -------------------------------
# 7. Define parallel chain
# -------------------------------
# RunnableParallel allows us to run multiple branches at once:
# - notes branch: (prompt_1 → model_2 → parser)
# - quiz branch: (prompt_2 → model_1 → parser)
# The output will be a dict: { "notes": ..., "quiz": ... }
parallel_chain = RunnableParallel({
    'notes': prompt_1 | model_2 | parser,
    'quiz': prompt_2 | model_1 | parser
})


# -------------------------------
# 8. Define merge chain
# -------------------------------
# Takes both "notes" and "quiz" as inputs, and merges them into one doc.
merge_chain = prompt_3 | model_1 | parser


# -------------------------------
# 9. Compose full chain
# -------------------------------
# Flow:
#   parallel_chain → produces {notes, quiz}
#   merge_chain → takes {notes, quiz} and merges into a final output
chain = parallel_chain | merge_chain


# -------------------------------
# 10. Run the chain
# -------------------------------
# Input is just {"text": text}, which is fed to both prompt_1 and prompt_2.
result = chain.invoke({'text': text})

# Print the merged result (notes + quiz in one document)
print(result)


# -------------------------------
# 11. Visualize the chain as ASCII graph
# -------------------------------
# This will print the execution graph (like a flow diagram).
chain.get_graph().print_ascii()