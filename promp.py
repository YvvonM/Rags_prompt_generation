import gradio as gr
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from extra import llm_model_cohere, llm_model_openai, PINECONE_INDEX_NAME
load_dotenv()

openai_api_key = os.getenv("OPEN_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")


embeddings = CohereEmbeddings(model = llm_model_cohere, cohere_api_key=cohere_api_key) #using the cohere embeddings
db = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
#saving conversations in memory
memory = ConversationBufferMemory(memory_key ='chat_history', 
                                  return_messages= False,)
#creating a prompt template
prompt_template = """
generate a better prompt from the user prompts. Make sure to use contexts from the text. If you do not know the answer to a question,
answer with "I do not know the answer, kindly ask another question or reframe your question."

Text: {context}
Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
#retriving the conversation
qa = ConversationalRetrievalChain.from_llm(
    llm = OpenAI(temperature=0, max_tokens=-1,openai_api_key=openai_api_key),
    chain_type = 'stuff',
    retriever = db.as_retriever(),
    memory=memory,
    get_chat_history= lambda h: h,
    verbose = True,
)

import gradio as gr

with gr.Blocks() as demo:
    # Define a Chatbot component
    chatbot = gr.Chatbot([], elem_id='chatbot').style(height=500)
    
    # Define a Textbox component
    msg = gr.Textbox()
    
    # Define a Button component for clearing
    clear = gr.Button("Clear")

    # Define a function for handling user input
    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    # Define a function for handling bot response
    def bot(history):
        print(history)
        # Assuming qa.run is a function that takes a dictionary and returns a response
        bot_message = qa.run({'question': history[-1][0], 'chat_history': history[:-1]})
        history[-1][1] = bot_message
        return history
    
    # Define the interaction flow
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio interface
demo.launch()
