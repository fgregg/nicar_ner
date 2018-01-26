import nltk
import collections
import pprint

def named_entities(text):
    entities = {'person' : collections.defaultdict(int),
                'organization' : collections.defaultdict(int),
                'geopolitical entity' : collections.defaultdict(int)}
                
    for sentence in nltk.sent_tokenize(text):
        tokenized_sentence = nltk.word_tokenize(sentence)
        tagged_sentence = nltk.pos_tag(tokenized_sentence)
        chunked_sentence = nltk.ne_chunk(tagged_sentence)
        for token in chunked_sentence:
            if hasattr(token, 'label'):
                if token.label() == 'PERSON':
                    type = 'person'
                elif token.label() == 'ORGANIZATION':
                    type = 'organization'
                elif token.label() == 'GPE':
                    type = 'geopolitical entity'

                name = ' '.join(word for word, pos in token.leaves())

                entities[type][name] += 1

    return entities
        

with open('example.txt') as f:
    entities = named_entities(f.read())
    pprint.pprint(entities)
