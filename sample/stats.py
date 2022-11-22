import os
import random

directory = "./dataset/HumanML3D/texts/"
NUM_SAMPLES = 1

contents = []

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        content = open(os.path.join(directory, filename)).read()
        if "sideways" in content:
            contents.append(content)
    else:
        continue
print(len(contents))
sample = random.sample(contents, NUM_SAMPLES)

print(sample)
