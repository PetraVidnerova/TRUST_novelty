# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# model_name = "mistralai/Mistral-Nemo-Instruct-2407"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     torch_dtype="auto"
# )

# llm = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=2048,
#     temperature=0.1
# )

# print(llm)


# template="""
#     Write me a greeting. 
# """



# res = llm(template)
# print(res)


# from langchain_huggingface import HuggingFacePipeline


# model_name = "mistralai/Mistral-Nemo-Instruct-2407"

# llm = HuggingFacePipeline.from_model_id(
#     model_id=model_name,
#     task="text-generation",
#     device_map="auto",
#     pipeline_kwargs={"max_new_tokens": 2048}
# )


# res = llm.invoke("What is Hugging Face?")

# print(res)

import pymupdf.layout 
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFacePipeline

pdf_path = "../../data/arxiv/PDF/arxiv_2405.11532v1.pdf"

md = pymupdf4llm.to_markdown(pdf_path)

#print(md)
headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
docs = splitter.split_text(md)

for d in docs:
    print("=================================================")
    print(d.page_content)




model_name = "mistralai/Mistral-Nemo-Instruct-2407"

llm = HuggingFacePipeline.from_model_id(
     model_id=model_name,
     task="text-generation",
     device_map="auto",
    pipeline_kwargs={"max_new_tokens": 2048}
)

for d in docs:
    num_tokens = len(llm.pipeline.tokenizer.encode(d.page_content))
    print("Tokens:", num_tokens)

print("Full tokens:",  len(llm.pipeline.tokenizer.encode(md)))   

context_length = llm.pipeline.model.config.max_position_embeddings
print("Context window:", context_length)

