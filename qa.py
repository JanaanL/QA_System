
import sys
import numpy as np
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity   

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


#def keyword_distance(keywords, candidates, story):
#    for keyword in keywords:
#        for candidate in candidates:
#            doc = nlp(storY)
#            for token in doc:
#                if token.text == candidate

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
    {<question_id>: {"Question":<question_string>, "Answer":<answer_string>, "Embeddings":<embedding vector for question>}}
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
                input_data = [ line.strip() for line in f ]
                input_data = list(filter(None, input_data))
                
                #if the document is a story, the entire text is saved in a dictionary along with a list of individuals sentences and their bert embeddings
                if file == ".story":
                    story_info = {"Text":None, "Sentences":None, "Embeddings":None, "NLP":None}
                    story_id = input_data[2].replace("STORYID: ","")
                    story = " ".join(input_data[4:])
                    story_info["Text"] = story
                    doc = nlp(story)
                    sentences = [sent.text for sent in doc.sents]
                    story_info["Sentences"] = sentences
                    story_info["NLP"] = doc
                    embeddings = []
                    for sentence in sentences:
                        embedding = get_bert_embedding(sentence, tokenizer, bert_model)
                        embeddings.append(embedding)
                    story_info["Embeddings"] = embeddings
                    story_dict[story_id] = story_info
                if file == ".answers":
                    for item in input_data:
                        if "QuestionID" in item:
                            question_id = item.replace("QuestionID: ","")
                            qa_info = {"Question":None, "Answer":None, "Embedding":None}
                        elif "Question" in item:
                            question = item.replace("Question: ","")
                            qa_info["Question"] = question
                            embedding = get_bert_embedding(question, tokenizer, bert_model)
                            qa_info["Embedding"] = embedding
                        elif "Answer" in item:
                            answer = item.replace("Answer: ","")
                            qa_info["Answer"] = answer
                            qa_dict[question_id] = qa_info        
    
    return story_dict, qa_dict

def create_answers(story_dict, qa_dict, nlp):
    for question_id, question_info in qa_dict.items():
        question = question_info["Question"]
        story_id = question_id[:10]
        question_embedding = question_info["Embedding"]
        answer = question_info["Answer"]
        candidates = set()
        keywords = set()
        doc = nlp(story_dict[story_id]["Text"])

        #WHERE questions
        if "where" in question.lower():
            for ent in doc.ents:
                if ent.label_ in ["LOC", "FAC"]:
                    candidates.add(ent.text)
            for chunk in doc.noun_chunks:
                if chunk.root.head.text in ["in","at","on"]:
                    candidates.add(chunk.text)
                    print(chunk.i)
        #WHO questions        
        elif "who" in question.lower():
            candidates_pos = {}
            for token in nlp(question):
                print(token.text, token.dep_, token.head.text)
            for ent in doc.ents:
                if ent.label_ in ["GPE","NORP","ORG","PERSON"]:
                    candidates.add(ent.text)
                    print(token.i)
            for token in nlp(question):
                if token.dep_ in ["dobj", "attr"]:
                    keywords.add(token.text)
            
        
        #WHEN questions
        elif "when" in question.lower():
            for ent in doc.ents:
                if ent.label_ in["DATE","EVENT","TIME"]:
                    candidates.add(ent.text)

        #HOW MANY
        elif "how many" in question.lower():
            for ent in doc.ents:
                if ent.label_ in ["CARDINAL","ORDINAL","PERCENT","QUANTITY"]:
                    candidates.add(ent.text)

        #HOW MUCH
        elif "how much" in question.lower():
            for ent in doc.ents:
                if ent.label_ in ["MONEY","PERCENT","QUANTITY"]:
                    candidates.add(ent.text)

        max_score = 0
        best_candidate = None
        for candidate in candidates:
            for index, sentence in enumerate(story_dict[story_id]["Sentences"]):
                if candidate in sentence:
                    sentence_embedding = story_dict[story_id]["Embeddings"][index]
                    similarity_score = cosine_similarity(question_embedding, sentence_embedding)
                    #keywords_score = compute_keywords_dist(keywords, story)
                    if similarity_score > max_score:
                        max_score = similarity_score
                        best_candidate = candidate
        print("Question: {}".format(question))            
        print("Best candidate: {}".format(best_candidate))
        print("Answer: {}".format(answer))


if __name__ == '__main__':

    input_file = sys.argv[1]
    nlp = spacy.load("en_core_web_sm")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    
    #open file for processing
    story_dict, qa_dict = process_data(input_file, nlp, tokenizer, bert_model)
    #print(story_dict, qa_dict)
    create_answers(story_dict, qa_dict, nlp)







    
