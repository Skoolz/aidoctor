import streamlit as st

import os

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import PromptTemplate

from langchain import OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
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
import time
import pandas as pd
import asyncio

import plotly.express as px

from langchain.prompts import SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate,ChatPromptTemplate

from langchain import text_splitter

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

sites = {
    'drugs':'drugs.com',
    'RxList':'rxlist.com'
}

st.set_page_config(page_title="Autonomous doctor", page_icon="💗")

st.header('Autonomous doctor 💗 is a program that helps a doctor choose the best treatment for a patient')

#with open('style.css') as f:
 #   st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)


choosed_site = ''
with st.sidebar:
    choosed_site = st.selectbox('Choose website',('drugs','RxList'),index=0)
    n_a = st.slider('Number of agents',min_value=2,max_value=10,value=4)
    st.session_state['n_a'] = n_a


search = DuckDuckGoSearchAPIWrapper()

max_tokens = 2500

def get_info_about_drug(text):
    res = search.run(f'site:{sites[str(choosed_site)]} {text} Warnings')
    s = text_splitter.TokenTextSplitter(chunk_size=max_tokens)
    return f'{sites[str(choosed_site)]} SOURCE: ' + s.split_text(res)[0]

#s = st.text_input('Drug name:')
#if(s):
#    st.write(get_info_about_drug(s))

def update():
    st.experimental_rerun()



df = pd.DataFrame({'Drug`s name':[]})
df['Drug`s name'] = df['Drug`s name'].astype(str)



async def aget_drug_desc(drug_infos,max_num=3):
    drug_info_template = """
    You are a doctor, and you need to briefly describe the drug:

    Use the following format:

    Input description: full description of the drug (what you get)

    Description: brief description of the drug (2-5 sentences)

    Contraindications: If you find any contraindications in the "Input description", briefly describe them. (Write only what is explicitly stated in the description, no need to write out indirect signs)
    
    Begin!

    Input description:{drug_info}

"""

    prompt = PromptTemplate(input_variables=['drug_info'],template=drug_info_template)

    llm = OpenAI(model='text-davinci-003',temperature=0.1)


    chain = LLMChain(prompt=prompt,llm=llm)
    descs = []
    drug_names = [d[0] for d in drug_infos]
    while(len(drug_infos)>0):
        count = min(len(drug_infos),max_num)
        tasks = [chain.arun(drug_info=drug_infos[i][1]) for i in range(count)]

        descs+=await asyncio.gather(*tasks)

        drug_infos = drug_infos[count:]
    return list(zip(drug_names,descs))

ratings = {
    'good choice':1.5,
    'can be used':1,
    'contraindicated':0.1,
    'uknown':0
}

async def aagent_get_result(drug_descs,patient_desc,max_num=3):
    template = """

    You are a doctor and your task is to evaluate the suitability of a given medication for a specific patient. You are provided with the patient's description, including their age, sex, and medical history, as well as the drug's information, including its description, uses, and contraindications.

    Please provide your conclusion in the following structured format:

    Patient Description:

    Drug Information:

    Thoughts: Before the conclusion, write down your thoughts
    Your thoughts should be like this. Based on the description of the patient and the drug information, we can conclude that there is (no) connection between them

    Conclusion:
    Based on your thoughts, choose one of three options: Contraindicated,Can be used,Good choice
    Use can be used if its use does not hurt much
    Use good choice if there are no connection
    In the Conclusion, write only what you have chosen. Nothing more.

    Example:
    Patient Description:
    suffers from hypertension, has COPD

    Drug Information:
    Description:
    Metoprolol is a beta-blocker that affects the heart and circulation (blood flow through arteries and veins).
    Metoprolol is used to treat angina (chest pain) and hypertension (high blood pressure).
    Metoprolol is also used to lower your risk of death or needing to be hospitalized for heart failure.
    Metoprolol injection is used during the early phase of a heart attack to lower the risk of death.
    Contraindications:asthma, chronic obstructive pulmonary disease (COPD), sleep apnea, or other breathing disorder

    Thoughts: Based on the description of the patient and the drug information, we can conclude that there is connection between them. The description of the patient says that he has COPD, the contraindications to the drug also mention COPD. Therefore, it is undesirable to use this drug.
    Conclusion: Contraindicated



    Begin!

    Patient Description:
    {patient_desc}

    Drug Information:
    {drug_desc}
    """

    prompt = PromptTemplate(template=template,input_variables=['patient_desc','drug_desc'])
    llm = ChatOpenAI(temperature=0.05,model='gpt-3.5-turbo')

    system_message = "You are a professional AI doctor's assistant."

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message)
    human_template=template
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=llm,prompt=chat_prompt)
    drug_names = [d[0] for d in drug_descs]
    recoms = []
    pattern = r'Conclusion:(.+)'
    while(len(drug_descs)>0):
        count = min(len(drug_descs),max_num)
        tasks = [chain.arun(patient_desc=patient_desc,drug_desc=drug_descs[i][1]) for i in range(count)]

        temp=await asyncio.gather(*tasks)
        for t in temp:
            match = re.search(pattern,t)
            if(match):
                var = match.group(1)
                var = var.lower()
                for rating_name in ratings:
                    if(var in rating_name):
                        var = rating_name
                        break
                    if(rating_name in var):
                        var = rating_name
                        break
                if(var in ratings):
                    recoms.append(var)
                else:
                    recoms.append('uknown')



        drug_descs = drug_descs[count:]
    return list(zip(drug_names,recoms))

def conver_rating_to_num(rating):
    return ratings[rating]

async def aagents_get_result(drug_descs,patient_desc,num_of_agents=3,max_num=3):
    res = []
    for n in range(num_of_agents):
        res.append(await aagent_get_result(drug_descs,patient_desc,max_num))
        st.write(f'agent {n} is finished')

    
    data = [pair for sublist in res for pair in sublist]

    df = pd.DataFrame(data=data,columns=['drug','rate_str'])
    st.session_state['rate_df'] = df

    df['rate_num'] = df['rate_str'].apply(conver_rating_to_num)
    
    return df


def get_drug_infos(drug_names):
    infos = []
    for name in drug_names:
        infos.append(get_info_about_drug(name))
    return list(zip(drug_names,infos))

def get_best_drug(df):
    df = df.pivot_table(index='drug',values='rate_num',aggfunc='sum')
    df = df.reset_index()
    return df.loc[df['rate_num'].idxmax(),'drug']



with st.form('form') as form:
    data = st.data_editor(df,num_rows='dynamic',width=300,height=300)
    

    patient_desc = st.text_area('Patient description:',placeholder='e.g. Suffers from hypertension, has COPD')

    btn = st.form_submit_button('Apply')
    if(btn):
        
        with st.spinner('Loading drug`s info'):
            drug_infos = get_drug_infos(list(data['Drug`s name']))
            st.session_state['infos'] = drug_infos
        st.success(body=' Step 1/3',icon='✅')
        infos = st.session_state.get('infos',[])
        #st.write(infos[0][1])
        with st.spinner('Calculating descriptions') as spin:
            infos = st.session_state.get('infos',[])
            st.session_state['descs'] = asyncio.run(aget_drug_desc(infos,3))
        st.success(body=' Step 2/3',icon='✅')
        st.write(st.session_state['descs'])
        with st.spinner('Calculating recomendations') as spin:
            descs = st.session_state.get('descs',[])
            num_of_agents = st.session_state.get('n_a',3)
            st.session_state['recoms'] = asyncio.run(aagents_get_result(descs,patient_desc,num_of_agents))
        st.success(body=' Step 3/3', icon='✅')
        data = st.session_state['recoms']
        data = data.pivot_table(index='drug',values='rate_num',aggfunc='sum')
        data = data.reset_index()
        data['color'] = [str(i) for i in data.index]
        fig = px.bar(orientation='h',data_frame=data,x='rate_num',y='drug',title='Drug evaluations',color='color')
        best_drug = get_best_drug(st.session_state['recoms'])
        print(best_drug)
        text = f'Based on the description of the patient: {str(best_drug)}'
        st.info(body=text,icon='🤖')
        st.info(body='This program cannot be used for self-medication! Consult a specialist!',icon='🚨')
        st.plotly_chart(fig)

