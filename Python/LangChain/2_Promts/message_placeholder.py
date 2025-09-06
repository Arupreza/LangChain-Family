from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template (modern syntax)
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# load chat history as a list of strings
chat_history = []
with open("C:/Users/rezan/Arupreza/LangChain/2_Promts/chat_history.txt", "r", encoding="utf-8") as f:
    chat_history.extend(f.readlines())

print("Chat history:", chat_history)

#create prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query': "Where is my refund?"})
print(prompt)