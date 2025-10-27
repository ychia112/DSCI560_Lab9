from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline, LlamaCpp
#from langchain.memory import ConversationBufferMemory
#from langchain.chains import ConversationalRetrievalChain
from config import DEFAULT_LLM, TOP_K

def build_llm(model_choice=DEFAULT_LLM):
    print(f"Initializing model backend: {model_choice}")
    
    if model_choice == 'openai':
        llm = ChatOpenAI(temperature=0.2)
        
    elif model_choice == "huggingface":
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-base",
            task="text2text-generation",
            model_kwargs={
                "temperature": 0.2,
                "repetition_penalty": 1.1
            },
            pipeline_kwargs={
                "max_new_tokens": 256
            }
        )
        
    elif model_choice == "llama":
        llm = LlamaCpp(
            model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
            n_ctx=4096,
            n_batch=512,
            temperature=0.2,
            repeat_penalty=1.1,
            verbose=False,
        )
        
    else:
        raise ValueError(f"Invalid model_choice: {model_choice}. Choose from 'openai', 'huggingface', 'llama'.")
    
    return llm

def run_conversation(vectorstore, model_choice=DEFAULT_LLM):
    llm = build_llm(model_choice)
    print(f"Chatbot ready using {model_choice.upper()}! Type 'exit' to quit.\n")
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vectorstore.as_retreiver(search_type="similarity", search_kwargs={"k": 4})#,
    #     #memory=memory
    # )
    
    chat_history = []
    
    while True:
        query = input("> ").strip()
        if query.lower() == "exit":
            print("Exiting chatbot.")
            break
        
        # Retrieve most relevant chunks
        docs = vectorstore.similarity_search(query, k=TOP_K)
        if not docs:
            print("No relevant documents found.")
            continue
        
        #context = "\n\n".join([d.page_content for d in docs])
        # Limit to top 3 chunks and truncate each to avoid token overflow
        context = "\n\n".join([d.page_content[:800] for d in docs[:3]])
        
        # Build prompt manually
        prompt = f"""
        You are a helpful assistant that answers based only on the provided context.
        
        Context:
        {context}
        
        Conversation so far:
        {chat_history[-3:] if chat_history else 'N/A'}
        
        Question: {query}
        Answer:
        """
        
        # Generate model response
        try:
            response = llm.invoke(prompt)
            answer = response.content.strip() if hasattr(response, "content") else str(response)
        except Exception as e:
            answer = f"Model error: {e}"
            
        # Print and store conversation
        print("\nBot:", answer, "\n" + "-" * 60)
        chat_history.append((query, answer))