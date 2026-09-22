import random
import pandas as pd
import voc as vc
from pathlib import Path

sentences = []

def apply_gram(subject, verb, obj=""):
    tense = random.choice(["", "past", "future"])
    negative = random.choice([True, False])

    if negative:
        sentence = f"{subject} not {verb}"
    else:
        sentence = f"{subject} {verb}"

    if obj:
        sentence += f" {obj}"
    if tense:
        sentence += f" {tense}"

    return sentence.strip()


for i in range(2000):
    subject = random.choice(vc.VOCABULARY["subjects"])
    verb = random.choice(list(vc.VERB_OBJECTS.keys()))
    obj = random.choice(vc.VERB_OBJECTS[verb])

    sentence = apply_gram(subject, verb, obj)
    sentences.append(sentence)


    verb = random.choice(vc.NO_OBJECT_VERBS)
    sentence = apply_gram(subject, verb)

    sentences.append(sentence)

    verb = random.choice(vc.MOVEMENT_VERBS)
    place = random.choice(vc.VOCABULARY["places"])
    sentence = apply_gram(subject, verb, place)


    sentences.append(sentence)


Path("datasets").mkdir(exist_ok=True)
df = pd.DataFrame({"text": sentences})
df = df.drop_duplicates()
df.to_csv("datasets/v1.csv", index=False)

print(df.head(10))
print()
print(f"Lenght - {len(df)}")
