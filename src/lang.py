# --------------------------------------------------------------
# Import Modules
# --------------------------------------------------------------

import os
import nest_asyncio
# import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.smith import RunEvalConfig, run_on_dataset

nest_asyncio.apply()

# --------------------------------------------------------------
# Load API Keys From the .env File
# --------------------------------------------------------------

load_dotenv(find_dotenv())
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "test"


# --------------------------------------------------------------
# LangSmith Quick Start
# --------------------------------------------------------------

client = Client()

llm = ChatOpenAI()
#llm.invoke("What can you do?")



example_inputs = [
    "a rap battle between Atticus Finch and Cicero",
    "a rap battle between Barbie and Oppenheimer",
    "a Pythonic rap battle between two swallows: one European and one African",
    "a rap battle between Aubrey Plaza and Stephen Colbert",
]

dataset_name = "Rap Battle Dataset"

# Storing inputs in a dataset lets us
# run chains and LLMs over a shared set of examples.

# dataset = client.create_dataset(
#     dataset_name=dataset_name,
#     description="Rap battle prompts.",
# )

#for input_prompt in example_inputs:
    # Each example must be unique and have inputs defined.
    # Outputs are optional
#    client.create_example(
#        inputs={"question": input_prompt},
#        outputs=None,
#        dataset_id=dataset.id,
#    )

# --------------------------------------------------------------
# 2. Evaluate Datasets with LLM
# --------------------------------------------------------------

eval_config = RunEvalConfig(
    evaluators=[
        # You can specify an evaluator by name/enum.
        # In this case, the default criterion is "helpfulness"
        "criteria",
        # Or you can configure the evaluator
        RunEvalConfig.Criteria("harmfulness"),
        RunEvalConfig.Criteria("misogyny"),
        RunEvalConfig.Criteria(
            {
                "cliche": "Are the lyrics cliche? "
                "Respond Y if they are, N if they're entirely unique."
            }
        ),
    ]
)

run_on_dataset(
    client=client,
    dataset_name=dataset_name,
    llm_or_chain_factory=llm,
    evaluation=eval_config,
)

# --------------------------------------------------------------
# 1. Create a Dataset From a List of Examples (Key-Value Pairs)
# --------------------------------------------------------------

example_inputs = [
    ("What is the largest mammal?", "The blue whale"),
    ("What do mammals and birds have in common?", "They are both warm-blooded"),
    ("What are reptiles known for?", "Having scales"),
    (
        "What's the main characteristic of amphibians?",
        "They live both in water and on land",
    ),
]

dataset_name = "Elementary Animal Questions"

#dataset = client.create_dataset(
#    dataset_name=dataset_name,
#    description="Questions and answers about animal phylogenetics.",
#)

#for input_prompt, output_answer in example_inputs:
#    client.create_example(
#        inputs={"question": input_prompt},
#        outputs={"answer": output_answer},
#        dataset_id=dataset.id,
#    )

# --------------------------------------------------------------
# 2. Evaluate Datasets With Customized Criterias
# --------------------------------------------------------------

evaluation_config = RunEvalConfig(
    evaluators=[
        # You can define an arbitrary criterion as a key: value
        #pair in the criteria dict
        RunEvalConfig.LabeledCriteria(
            {
                "smart": (
                    "Can the submission be considered smart,"
                    "this should constitute 50% of the final score"
                ),
                "too long": (
                    "is the number of characters in the response over 100?"
                    "this should constitute 50% of the final score"
                )
            }
        ),
    ]
)

run_on_dataset(
    client=client,
    dataset_name="Elementary Animal Questions",
    llm_or_chain_factory=llm,
    evaluation=evaluation_config,
)
