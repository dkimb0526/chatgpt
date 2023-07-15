import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain


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
    template="give me a quick summary of the book in 30 words{title}",
)

# llms
llm = OpenAI(temperature=0.9)
# chain
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key="title")
script_chain = LLMChain(
    llm=llm, prompt=script_template, verbose=True, output_key="script"
)
sequential_chain = SequentialChain(
    chains=[title_chain, script_chain],
    input_variables=["topic"],
    output_variables=["title", "script"],
    verbose=True,
)

# Get response from input
if prompt:
    response = sequential_chain({"topic": prompt})
    st.write(response["title"])
    st.write(response["script"])
