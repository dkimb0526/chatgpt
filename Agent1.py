import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


os.environ["OPENAI_API_KEY"] = apikey

# APP FrameWork
st.title("🦜️🔗Agent1")
# Input var
prompt = st.text_input("Hello sir, how many I help you?")

# prompt templates
title_template = PromptTemplate(
    input_variables=["topic"],
    template=" {topic}",
)

# templates2
script_template = PromptTemplate(
    input_variables=["title"],
    template="{title}",
)
# memory
title_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
script_memory = ConversationBufferMemory(input_key="title", memory_key="chat_history")

# llms
llm = OpenAI(temperature=0.9)
# chain
title_chain = LLMChain(
    llm=llm,
    prompt=title_template,
    verbose=True,
    output_key="title",
    memory=title_memory,
)
script_chain = LLMChain(
    llm=llm,
    prompt=script_template,
    verbose=True,
    output_key="script",
    memory=script_memory,
)

wiki = WikipediaAPIWrapper()

# sequential_chain = SequentialChain(
#    chains=[title_chain, script_chain],
#    input_variables=["topic"],
#    output_variables=["title", "script"],
#    verbose=True,
# )

# Get response from input
if prompt:
    # response = sequential_chain({"topic": prompt})
    # st.write(response["title"])
    # st.write(response["script"])
    title = title_chain.run(topic=prompt)
    wiki_research = wiki.run(prompt)
    # script = script_chain.run(title=title, wikipedia_research=wiki_research)
    script = script_chain.run(title=title)
    st.write(title)
    st.write(script)

    with st.expander("Title history"):
        st.info(title_memory.buffer)

    with st.expander("Script history"):
        st.info(script_memory.buffer)

    with st.expander("Wikipedia Research"):
        st.info(wiki_research)
