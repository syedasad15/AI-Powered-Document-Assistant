from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI  # Updated import
from prompts import custom_qa_prompt  # ✅ Corrected import

from openai import OpenAIError

def create_qa_chain(vector_store):
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_tokens=512
        )
        print("✅ ChatOpenAI initialized.")
    except OpenAIError as e:
        print(f"❌ OpenAI initialization failed: {e}")
        raise
    retriever = vector_store.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.7})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_qa_prompt}
    )
    return qa_chain