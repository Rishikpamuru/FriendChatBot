from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import pandas as pd
model_name = "deepset/roberta-base-squad2"

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
df = pd.read_csv("Messages.csv")
friendname = "name"
context = df
        


while True:
    user_input = input("Hey my name is " + friendname + " ask me a question!: ")

    if user_input.lower() == 'q':
        break

    QA_input = {
        'question': user_input,
        'context': context
    }

    res = nlp(QA_input)
    print(res)

    print(f"Answer: {res['answer']}") 