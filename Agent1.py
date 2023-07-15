import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain


os.environ["OPENAI_API_KEY"] = apikey

# APP FrameWork
st.title("🦜️🔗Agent1")
# Input var
prompt = st.text_input("Hello sir, how many I help you?")

# prompt templates
title_template = PromptTemplate(
    input_variables=["topic"],
    template="give a me title on a book about {topic}",
)

# templates2
script_template = PromptTemplate(
    input_variables=["title"],
    template="give me a quick summary of the book {title}",
)

# llms
llm = OpenAI(temperature=0.9)
# chain
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)
sequential_chain = SimpleSequentialChain(
    chains=[title_chain, script_chain], verbose=True
)

# Get response from input
if prompt:
    response = sequential_chain.run(prompt)
    st.write(response)
