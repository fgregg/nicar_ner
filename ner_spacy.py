import spacy
import collections
import pprint

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en')

# Process whole documents
text = open('example.txt').read()
doc = nlp(text)

entities = {'person' : collections.defaultdict(int),
            'organization' : collections.defaultdict(int),
            'geopolitical entity' : collections.defaultdict(int)}


# Find named entities, phrases and concepts
for entity in doc.ents:
    if entity.label_ == 'PERSON':
        type = 'person'
    elif entity.label_ == 'ORG':
        type = 'organization'
    elif entity.label_ == 'GPE':
        type = 'geopolitical entity'
    else:
        print(entity.label_)
        continue

    entities[type][entity.text] += 1

pprint.pprint(entities)
