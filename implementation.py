from PIL import Image
import requests
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import torch
import nltk
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from random import randint
import nltk.data
import random

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

accuracy_default = 0.0
accuracy_synonym = 0.0
accuracy_hypernym = 0.0
accuracy_hyponym = 0.0
accuracy_paraphrase = 0.0

def loadDataset(setSize=10):
  ds = load_dataset("nlphuji/flickr30k", split="test")
  print(len(ds))

  dataset = ds.filter(lambda x : x["split"] == "test")
  print(len(dataset))
  dataset = dataset.shuffle(seed=50).select(range(setSize))
  images = [x['image'] for x in dataset]
  captions = [x['caption'][0] for x in dataset]
  map = range(setSize)

  return images, captions, map

def getRecall(images, captions, textToImageMap, top):
  setSize = len(captions)
  model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
  processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

  input_images = processor(images=images, return_tensors="pt")
  input_texts = processor(
      text=captions, return_tensors="pt", padding=True, truncation=True, max_length=77
  )

  with torch.inference_mode():
      image_features = model.get_image_features(**input_images)
      text_features = model.get_text_features(**input_texts)

  image_features = image_features/image_features.norm(dim=-1, keepdim=True)
  text_features = text_features/text_features.norm(dim=-1, keepdim=True)
  similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)

  truePredic = 0

  for i in range(setSize):
    values, indices = similarity[i].topk(top)
    # print(values)
    if textToImageMap[i] in indices:
      truePredic += 1
  recall = truePredic/setSize
  return recall

def replaceWithSynonym(text, replaceRate):
  def getPOS(pos):
    if pos[0] == 'N':
      return wn.NOUN
    if pos[0] == 'V':
      return wn.VERB
    if pos[0] == 'J':
      return wn.ADJ

  output = ""
  tokenized = tokenizer.tokenize(text)
  words = word_tokenize(text)
  tagged = nltk.pos_tag(words)

  for i in range(0,len(tagged)):
      change = True

      if tagged[i][1] == 'NNP' or tagged[i][1] == 'DT' or (tagged[i][1][0] not in ['N', 'V', 'J']) or random.random() > replaceRate:
        change = False

      word = tagged[i][0]
      pos = tagged[i][1]

      synsets = wordnet.synsets(word, pos=getPOS(pos))
      arr = [synset.lemma_names() for synset in synsets]
      synonyms = []
      for x in arr:
        for y in x:
          if y.lower() != word.lower():
            synonyms.append(y)
      random.shuffle(synonyms)
      if len(synonyms) > 0:
        synonym = synonyms[0]
      else:
        change = False
      output += (synonym if change else word) + " "
  return output

def replaceWithHyperHyponym(text, replaceRate, hyper=True):
  def getPOS(pos):
    if pos[0] == 'N':
      return wn.NOUN
    if pos[0] == 'V':
      return wn.VERB
    if pos[0] == 'J':
      return wn.ADJ

  output = ""
  tokenized = tokenizer.tokenize(text)
  words = word_tokenize(text)
  tagged = nltk.pos_tag(words)

  for i in range(0,len(tagged)):
      change = True

      if tagged[i][1] == 'NNP' or tagged[i][1] == 'DT' or (tagged[i][1][0] not in ['N', 'V', 'J']) or random.random() > replaceRate:
        change = False
      # if tagged[i][1] == 'NNP' or tagged[i][1] == 'DT' or (tagged[i][1][0] not in ['N']) or random.random() > replaceRate:
      #   change = False

      word = tagged[i][0]
      pos = tagged[i][1]

      synsets = wordnet.synsets(word, pos=getPOS(pos))
      hypnyms = []

      for synset in synsets:
        if word.lower() in synset.lemma_names():
          hypnyms += synset.hypernyms() if hyper else synset.hyponyms()

      arr = [synset.lemma_names() for synset in hypnyms]

      synonyms = []
      for x in arr:
        for y in x:
          if y.lower() != word.lower():
            synonyms.append(y)
      random.shuffle(synonyms)
      if len(synonyms) > 0:
        synonym = synonyms[0]
      else:
        change = False
      output += (synonym if change else word) + " "
  return output

device = "cuda"
device = "cpu"
paraphraseTokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
paraphraseModel = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to(device)

def paraphrase(sentence, count):
  text =  "paraphrase: " + sentence + " </s>"

  encoding = paraphraseTokenizer.encode_plus(text, return_tensors="pt")

  input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

  outputs = paraphraseModel.generate(
      input_ids=input_ids, attention_mask=attention_masks,
      max_length=77,
      do_sample=True,
      top_k=5,
      top_p=0.95,
      early_stopping=True,
      temperature=1.5,
      num_return_sequences=count
  )
  final_outputs = []

  for output in outputs:
      line = paraphraseTokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
      final_outputs.append(line)
  return final_outputs

print(paraphrase("It is raining today and I will not go outside.", 5))


setSize = 100
recallK = 5
images, captions, textToImageMap = loadDataset(setSize)


print(captions[0])
print(replaceWithSynonym(captions[0], 0.5))
print(replaceWithHyperHyponym(captions[0], 0.5))
print(paraphrase(captions[0], 5))

accuracy_default = getRecall(images, captions, textToImageMap, recallK)
print("accuracy_default = ", accuracy_default)

attackCountPerImage = 5
attackedCaptions = []
attackedMap = []
for i in range(len(captions)):
  for j in range(attackCountPerImage):
    attackedCaptions.append(replaceWithSynonym(captions[i], 0.5))
    attackedMap.append(i)

accuracy_synonym = getRecall(images, attackedCaptions, attackedMap, recallK)
print("accuracy_synonym = ", accuracy_synonym)

attackCountPerImage = 5
attackedCaptions = []
attackedMap = []
for i in range(len(captions)):
  for j in range(attackCountPerImage):
    attackedCaptions.append(replaceWithHyperHyponym(captions[i], 0.5, False))
    attackedMap.append(i)

accuracy_hypernym = getRecall(images, attackedCaptions, attackedMap, recallK)
print("accuracy_hypernym = ", accuracy_hypernym)

attackCountPerImage = 5
attackedCaptions = []
attackedMap = []
for i in range(len(captions)):
  for j in range(attackCountPerImage):
    attackedCaptions.append(replaceWithHyperHyponym(captions[i], 0.5, True))
    attackedMap.append(i)

accuracy_hyponym = getRecall(images, attackedCaptions, attackedMap, recallK)
print("accuracy_hyponym = ", accuracy_hyponym)

attackCountPerImage = 5
attackedCaptions = []
attackedMap = []
for i in range(len(captions)):
  phrasedTexts = paraphrase(captions[i], attackCountPerImage)
  for j in range(attackCountPerImage):
    attackedCaptions.append(phrasedTexts[j])
    attackedMap.append(i)

accuracy_paraphrase = getRecall(images, attackedCaptions, attackedMap, recallK)
print("accuracy_paraphrase = ", accuracy_paraphrase)

import matplotlib.pyplot as plt
import numpy as np

attack_types = ["Actual", "Synonym", "Hypernym", "Hyponym", "Paraphrase"]

recall = [accuracy_default, accuracy_synonym, accuracy_hypernym, accuracy_hyponym, accuracy_paraphrase]
accuracy_drop = []
for i in range(len(recall)):
  accuracy_drop.append(accuracy_default - recall[i])

x = np.arange(len(attack_types))
width = 0.3

plt.figure()

plt.bar(x - width/2, recall, width, label="Recall@5")

plt.bar(x + width/2, accuracy_drop, width, label="Accuracy drop")

plt.xticks(x, attack_types)

plt.xlabel("Attack Type")
plt.ylabel("Value")
plt.title("Recall@5 vs Accuracy Drop by Attack type")
plt.legend()
plt.tight_layout()
plt.show()


print("accuracy_default = ", accuracy_default, "\t drop = ", accuracy_default-accuracy_default)
print("accuracy_synonym = ", accuracy_synonym, "\t drop = ", accuracy_default-accuracy_synonym)
print("accuracy_hypernym = ", accuracy_hypernym, "\t drop = ", accuracy_default-accuracy_hypernym)
print("accuracy_hyponym = ", accuracy_hyponym, "\t drop = ", accuracy_default-accuracy_hyponym)
print("accuracy_paraphrase = ", accuracy_paraphrase, "\t drop = ", accuracy_default-accuracy_paraphrase)


attackCountPerImage = 5
attackedCaptions = []
attackedMap = []

for i in range(len(captions)):
  for j in range(attackCountPerImage):
    attackedCaptions.append(replaceWithSynonym(captions[i], 0.5))
    attackedMap.append(i)

def cherry_pick(images, captions, attackedCaptions, attackCountPerImage, setSize, top):
  model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
  processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

  for i in range(setSize):
    myCaptions = []
    myCaptions.append(captions[i])
    for j in range(attackCountPerImage):
      myCaptions.append(attackedCaptions[i*5 + j])

    input_images = processor(images=images, return_tensors="pt")
    input_texts = processor(
        text=myCaptions, return_tensors="pt", padding=True, truncation=True, max_length=77
    )

    with torch.inference_mode():
        image_features = model.get_image_features(**input_images)
        text_features = model.get_text_features(**input_texts)

    image_features = image_features/image_features.norm(dim=-1, keepdim=True)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)


    for k in range(attackCountPerImage+1):
      values, indices = similarity[k].topk(top)
      # print(values)
      if i not in indices:
        if k == 0:
          continue
        else:
          print(myCaptions[0])
          print(myCaptions[k])
          print("====================================")

print(range(attackCountPerImage+1))
cherry_pick(images, captions, attackedCaptions, attackCountPerImage, setSize, recallK)

