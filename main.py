from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
from argparse import ArgumentParser

load_dotenv()

parser = ArgumentParser()
parser.add_argument("--language", type=str, required=True, default="python")
parser.add_argument("--task", type=str, required=True, default="return a list of numbers")
args = parser.parse_args()

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very short {language} function that will {task}"
)

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write code for unit test for the following {language} code:\n{code}"
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key='code')

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key='test'
)
chain = SequentialChain(chains=[code_chain, test_chain],
                        input_variables=["task", "language"],
                        output_variables=['test', 'code'])  # noqa
result = chain({
    "language": args.language,
    "task": args.task
})

print(result["code"])
print(result["test"])
