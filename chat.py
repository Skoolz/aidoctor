import streamlit as st

import os

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate

from langchain import OpenAI, LLMChain
from langchain.tools import DuckDuckGoSearchRun

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish,HumanMessage
import re
import langchain
from duckduckfix import DuckDuckGoSearchAPIWrapper
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import BaseChatPromptTemplate

from langchain.agents import initialize_agent
from langchain.agents import ConversationalChatAgent
from langchain.agents import AgentType

from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space


sites = {
    'webmd':'webmd.com',
    'medlineplus':'medlineplus.gov',
    'wiki':'en.wikipedia.org'
}





st.set_page_config(page_title="Autonomous researcher - Twitter threads", page_icon=":bird:")


choosed_site = ''
with st.sidebar:
    choosed_site = st.selectbox('Choose website',('webmd','medlineplus','wiki'),index=0)

search = DuckDuckGoSearchAPIWrapper()

def input_wrapper(text):
    return f'{str(choosed_site)} SOURCE: ' + search.run(f'site:{sites[str(choosed_site)]} {text}')

tools = [
    Tool(
        name = "Search",
        func=input_wrapper,
        description="useful for when you need to answer questions about medicine. Don't use the site name in input"
    )
]

from langchain.chat_models import ChatOpenAI



from langchain.chains.conversation.memory import ConversationBufferWindowMemory

@st.cache_resource
def model():
    llm=ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo'
    )
    memory=ConversationBufferWindowMemory(k=10,memory_key="chat_history", input_key='input', output_key="output",return_messages=True)
    conversational_agent = initialize_agent(tools=tools,llm=llm,agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,memory=memory,verbose=True,return_intermediate_steps=True)

    st.write(conversational_agent.agent.llm_chain.prompt.messages)
    return conversational_agent

model()

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm HugChat, How may I help you?"]

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']




# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
## Applying the user input box
with input_container:
    user_input = get_text()

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    agent = model()
    response = agent(prompt)
    st.session_state.memory = agent.memory.buffer
    return response['output']

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))

with st.expander('History'):
        st.info(model().memory)
#st.write(memory.chat_memory)