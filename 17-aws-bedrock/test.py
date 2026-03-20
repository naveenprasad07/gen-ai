from openai import OpenAI
import os

client = OpenAI()


prompt_data = " Describe about mother?"

response = client.chat.completions.create(
    model="openai.gpt-oss-120b", 
    messages=[
        {"role": "system", "content": "Act as a shakespeare and answer every questions as a poem"},
        {"role": "user", "content": prompt_data }
    ]
)

output = response.choices[0].message.content

with open("output.md","a") as file:
    file.write(output)
    file.write("\n")



print(output)
