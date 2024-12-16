#OpenAI's GPT Models

#GPT-3.5 (Ada, Babbage, Curie, Davinci):

#OpenAI offers GPT-3.5 models through their API. While it's not entirely free, there is a free tier with a limited amount of usage, making it a good starting point for experimentation.

import openai

openai.api_key = 'your-api-key'

response = openai.Completion.create(
  model="gpt-3.5-turbo",
  prompt="What is the capital of France?",
  max_tokens=10
)

print(response.choices[0].text.strip())


# Hugging Face Transformers (Free Models)

#Hugging Face provides a wide range of pre-trained models, including open-source language models such as GPT-2, BERT, T5, and others, which are available for free.

#pip install transformers
#pip install torch 
#pip install tensorflow
#pip install tensorflow-gpu

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

inputs = tokenizer("What is the capital of France?", return_tensors="pt")
outputs = model.generate(inputs['input_ids'], max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


#Google's T5 and BERT Models

#T5 (Text-to-Text Transfer Transformer) and BERT models are available for free via Hugging Face or directly from Google's repositories.

from transformers import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

inputs = tokenizer("translate English to French: What is your name?", return_tensors="pt")
outputs = model.generate(inputs['input_ids'])

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


#EleutherAI's GPT-Neo and GPT-J

#GPT-Neo and GPT-J are open-source models developed by EleutherAI. They are designed to be free alternatives to GPT-3 and can be used via Hugging Face.

from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs['input_ids'], max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


#BLOOM by BigScience

#BLOOM is a multilingual LLM developed by BigScience, and it's open-source. Available on Hugging Face, it supports more than 45 languages and has multiple model sizes.

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

inputs = tokenizer("Translate to French: Hello, how are you?", return_tensors="pt")
outputs = model.generate(inputs['input_ids'], max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


#import torch
#print(torch.__version__)




#youtube

#https://www.youtube.com/watch?v=QEaBAZQCtwE

from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for a HuggingFace course my whole life")

print(res)


