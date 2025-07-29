from langchain.memory import ConversationBufferWindowMemory

def get_conversation_memory(k=10):
    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=k,  # Remember last k exchanges
        output_key='answer'
    ) 