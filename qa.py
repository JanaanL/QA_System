
import sys
import numpy as np
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

'''
Takes a string of text, a bert tokenizer and a bert model and returns an embedding vector of size (1,768)
'''
def get_bert_embedding(text, tokenizer, bert_model):
    
    encoded_input = tokenizer.encode_plus(text, max_length=128, truncation=True, padding='max_length', return_tensors='pt')                                                
    output = bert_model(**encoded_input)
    embeddings = output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    return mean_pooled.detach().numpy()


def process_data(input_file, nlp, tokenizer, bert_model):
    
    '''
    Dictionary that contains all of the information for the story.
    The key is the story id and the value is another dictionary that contains the full text, a list of sentences,
    a list of their bert_encodings and the nlp information for the text corpus.
    {<story_id>: {"Text": <story_text_string>, "Sentences":{A list of strings that are the sentences in the corpus.  Listed in order], "Embeding":[list of embedding vectors],
    "NLP": <nlp info derived using Spacy>}}
    '''
    story_dict = {}

    '''
    Dictionary that contains the qestions and answers.  The key is the question_id and the values are a dictionary with the question text, answer text, embedding
    vector for the question.
    {<question_id>: {"Question":<question_string>, "Answer":<answer_string>, "Embedding":<embedding vector for question>}}
    '''
    qa_dict = {}

    with open(input_file) as f:
        contents = [ line.strip() for line in f ]
    directory = contents[0]
    for i in tqdm(range(1, len(contents))):
        item = contents[i]
        for file in [".story", ".questions", ".answers"]:
            path = directory + item + file
            with open(path) as f:
                
                #if the document is a story, the entire text is saved in a dictionary along with a list of individuals sentences and their bert embeddings
                if file == ".story":
                    story_info = {"Text":None, "Sentences":None, "Embedding":None, "NLP":None}
                    story = [ line.strip() for line in f ]
                    story = list(filter(None, story))
                    story = " ".join(story[4:])
                    story_info["Text"] = story
                    doc = nlp(story)
                    sentences = [sent.text for sent in doc.sents]
                    print(sentences)
                    print("\n")
                    story_info["Sentences"] = sentences
                    story_info["NLP"] = doc
                    embeddings = []
                    for sentence in sentences:
                        embedding = get_bert_embedding(sentence, tokenizer, bert_model)
                        embeddings.append(embedding)
                    story_info["Embedding"] = embeddings

                if file == ".answers":
                    qa_info = {"Question":none, "Answer":None, "Embedding":None}


if __name__ == '__main__':

    input_file = sys.argv[1]
    nlp = spacy.load("en_core_web_sm")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    
    #open file for processing
    data = process_data(input_file, nlp, tokenizer, bert_model)
    #unpack data
    #story_dict, qa_dict, signature_vectors = input_data

    #get_answers(story_dict, qa_dict, signature_vectors, nlp, tokenizer, bert_model)







    
