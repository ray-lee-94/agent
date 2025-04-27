# %%
from langchain_openai import ChatOpenAI

openai_api_key="EMPTY"
openai_api_base="http://192.168.0.67:1234/v1/"
llm = ChatOpenAI(openai_api_key=openai_api_key,openai_api_base=openai_api_base)

print(llm.invoke("Hello, world!").content)

# %%

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_template("请根据下面主题写一篇小红书营销短文: {topic}")

output_parser=StrOutputParser()

chain= prompt | llm  | output_parser

for chunk in chain.stream({"topic": "方便面"}):
    print(chunk,end="")
   
# %% 
from langchain_core.prompts import ChatPromptTemplate

chat_template= ChatPromptTemplate.from_messages(
    [
        ("system",""" 你是一直很粘人的猫猫，你的名字叫{name}。
         我是你的主人，你每天都有和我说不完的话，下面请开启我们的聊天要求：
         1. 你的语气要像一只猫，回答的过程可以夹一些语气词
         2. 你对生活的观察有很独特的视角，一些想法很难在人类身上看到
         3. 你的语气很可爱，既会认真倾听和我的对话，又会不断开启新话题
         下面从你迎接我下班回家开始开启我们今天的对话
         """),
        ("human","{user_input}")
    ]
)

message=chat_template.format_messages(name="喵喵",user_input="想我了吗？")
response=llm.invoke(message)
print(response.content)

# %%
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template.append(response)
chat_template.append(HumanMessage(content="今天我遇到一只老鼠"))

message=chat_template.format_messages(name="喵喵",user_input="想我了吗？")
response=llm.invoke(message)
print(response.content)
