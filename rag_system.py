# rag_system.py
# Entry point for the modular EnhancedRAGSystem

from rag_system import EnhancedRAGSystem

# Optionally, you can provide a simple usage example or CLI here
if __name__ == "__main__":
    rag = EnhancedRAGSystem()
    rag.initialize_components()
    rag.create_vectorstore("documents")
    rag.setup_conversational_chain()
    print("RAG system initialized and ready.")
    # Example interaction
    while True:
        question = input("Ask a question (or 'exit'): ")
        if question.lower() == 'exit':
            break
        response = rag.get_response(question)
        print("Answer:", response["answer"])
        print("Confidence:", response["confidence"])
        print("Sources:", response["sources"])
