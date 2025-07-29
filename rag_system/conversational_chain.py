from langchain.chains import ConversationalRetrievalChain
import logging

def setup_conversational_chain(llm, vectorstore, memory, custom_prompt):
    try:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}
            ),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            verbose=True
        )
        logging.info("Conversational chain setup successfully")
        return chain
    except Exception as e:
        logging.error(f"Error setting up conversational chain: {e}")
        return None 