import sys
import numpy as np
import spacy
import torch
import pickle
import re
#import argparse
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity   
from nltk.stem.lancaster import LancasterStemmer
from spacy import displacy
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

def find_position_of_phrase(phrase, doc):
    text = phrase.split()
    start_pos = 0
    cur_pos = 0
    text_index = 0
    for token in doc:
        if token.text == text[text_index]:
            if start_pos == 0:
                start_pos = token.i
            text_index += 1
            if text_index == len(text):
                break
        else:
            start_pos == 0
   
    #if start_pos == 0:
        #print("Could not find position of {} in document".format(phrase))
    return start_pos

def keyword_distance(keywords, candidate, doc):
    st = LancasterStemmer()
    can_start_pos = find_position_of_phrase(candidate, doc)
    keywords_dist = np.zeros(len(keywords))
    for token in doc:
        for index, keyword in enumerate(keywords):
            if st.stem(keyword) == st.stem(token.text.lower()):
                #print("Found a match for the keyword {}".format(st.stem(token.text.lower())))
                dist = abs(can_start_pos - token.i)
                keywords_dist[index] = dist
    average = np.mean(keywords_dist)
    if average != 0:
        return 1.0 / average
    else:
        return 0

def process_data(input_file, nlp, tokenizer, bert_model, process=True):

    if process:


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
        #for i in range(1, len(contents)):
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

#        filename = "story_dict"
#        outfile = open(filename,'wb')
#        pickle.dump(story_dict,outfile)
#        outfile.close()
#
#        filename = "qa_dict"
#        outfile = open(filename,'wb')
#        pickle.dump(qa_dict,outfile)
#        outfile.close()

    else:
        infile = open("story_dict",'rb')
        story_dict = pickle.load(infile)
        infile.close()
        
        infile = open("qa_dict",'rb')
        qa_dict = pickle.load(infile)
        infile.close()

    return story_dict, qa_dict

def create_answers(story_dict, qa_dict, nlp):
    st = LancasterStemmer()
    
    for question_id, question_info in qa_dict.items():
        question = question_info["Question"]
        story_id = question_id[:10]
        question_embedding = question_info["Embedding"]
        answer = question_info["Answer"]
        candidates = set()
        keywords = set()
        doc = nlp(story_dict[story_id]["Text"])
        q_tokens = nlp(question)
#        displacy.serve(ques, style='dep')

        #WHERE questions
#        if "where" in question.lower():
##            print(question)
#            for ent in doc.ents:
#                if ent.label_ in ["LOC", "FAC"]:
#                    candidates.add(ent.text)
#            for chunk in doc.noun_chunks:
#                for candidate in candidates.copy():
#                   if candidate in chunk.text:
#                       candidates.remove(candidate)
#                       candidates.add(chunk.text)
#                if chunk.root.head.text in ["in","at","on","by"]:
#                    candidates.add(chunk.text)
#            for q_token in q_tokens:
#                if q_token.dep_ in ["nsubj", "ROOT", "dobj", "pobj", "ccomp"]:
#                    keywords.add(q_token.text)
#       
#       #WHO questions        
#        elif  "who" in question.lower():
##            for token in nlp(question):
##                print(token.text, token.dep_, token.head.text)
#
#            for ent in doc.ents:
#                if ent.label_ in ["GPE","NORP","ORG","PERSON"]:
##                    print(ent.text, ent.label_)
#                    candidates.add(ent.text)
#            for chunk in doc.noun_chunks:
# #               print(chunk.text)
#                for candidate in candidates.copy():
#                    if candidate in chunk.text:
#                        candidates.remove(candidate)
#                        candidates.add(chunk.text)
#            #print("Candidates: {}".format(candidates))
#            for token in nlp(question):
#                if token.dep_ in ["dobj", "attr","nsubj", "pobj"]:
#                    keywords.add(token.text.lower())
#           
#        
#        #WHEN questions
#        elif question.startswith("When"):
#            from_to_candidate = re.search("^[Ff]+rom [\d\w\s,.]+ to [\d\w\s,.]+$", story_dict[story_id]["Text"])
#            if from_to_candidate is not None:
#                candidates.add(from_to_candidate.group())
#            for token in doc:
#                if token.text == "when":
#                    head = token.head
#                    phrase = ""
#                    for word in head.subtree:
#                        phrase = phrase + word.text + " "
#                    candidates.add(phrase)
#
#            for ent in doc.ents:
#                if ent.label_ in["DATE","EVENT","TIME"]:
#                    candidates.add(ent.text)
#            for chunk in doc.noun_chunks:
#                for candidate in candidates.copy():
#                    if candidate in chunk.text:
#                        candidates.remove(candidate)
#                        candidates.add(chunk.text)
#                for prep in ["during", "after", "before", "while", "as"]:
#                    if chunk.root.head.text == prep:
#                        candidates.add(prep + " " + chunk.text)
#            for token in nlp(question):
#                if token.dep_ in ["dobj", "attr","nsubj", "pobj","ROOT"]:
#                    keywords.add(token.text.lower())

        #HOW MANY
        if question.startswith("How"):
            second_word = question.split(' ')[1]
            if second_word == "many": 
                for ent in doc.ents:
                    if ent.label_ in ["CARDINAL","PERCENT","QUANTITY"]:
                        candidates.add(ent.text)
                for token in nlp(question):
                    if token.dep_ in ["dobj", "attr","nsubj", "pobj","ROOT"]:
                        keywords.add(token.text.lower())

            #HOW MUCH
            elif second_word == "much":
                for ent in doc.ents:
                    if ent.label_ in ["MONEY","PERCENT"]:
                        candidates.add(ent.text)
                for token in nlp(question):
                    if token.dep_ in ["dobj", "attr","nsubj", "pobj","ROOT"]:
                        keywords.add(token.text.lower())

            else:
                for token in nlp(question):
                    if token.text == second_word:
                        if token.pos_ in ["ADV", "ADJ"]:
                            for ent in doc.ents:
                                if ent.label_ in ["CARDINAL", "ORDINAL", "QUANTITY"]:
                                    candidates.add(ent.text)
                            for token in nlp(question):
                                if token.dep_ in ["dobj", "attr","nsubj", "pobj","ROOT"]:
                                    keywords.add(token.text.lower())   

                        if token.pos_ in ["AUX", "VERB"]:
                            for word in nlp(question):
                                if word.dep_ == "nsubj":
                                    subject = word.text
                                    print("Subject {}".format(subject))
                                    for w in doc:
                                        if w.text == subject:
                                            subtree = w.head.subtree
                                            phrase = ""
                                            for word in subtree:
                                                phrase = phrase + word.text + " "
                                            candidates.add(phrase)
                                            print("Phrase added: {}".format(phrase))
                                if word.dep_ == "ROOT":
                                    keywords.add(word.text)
                            for token in nlp(question):
                                if token.dep_ in ["dobj", "attr","nsubj", "pobj", "ROOT", "iobj"]:
                                    keywords.add(token.text.lower())

        
        elif "why" in question.lower() or "what" in question.lower():
            for token in nlp(question):
                if token.dep_ in ["dobj", "attr","nsubj", "pobj","ROOT"]:
                    keywords.add(token.text.lower())
            
        max_score = 0
        best_candidate = ""
        for candidate in candidates:
            print("Candidate for consideration: {}".format(candidate))
            for index, sentence in enumerate(story_dict[story_id]["Sentences"]):
                if candidate in sentence:
                    sentence_embedding = story_dict[story_id]["Embeddings"][index]
                    similarity_score = cosine_similarity(question_embedding, sentence_embedding)
                    if len(keywords) > 0:
                        keywords_score = keyword_distance(keywords, candidate, doc)
                    else:
                        keywords_score = 0
                    score = similarity_score + keywords_score

                    if score > max_score:
                        max_score = score
                        best_candidate = candidate
#        print("QuestionID: {}".format(question_id))
#        print("Answer: {}".format(best_candidate))            
#        print()
       #print("Answer Candidate: {}".format(best_candidate))
        if question.startswith("How"):
            print("Question: {}".format(question))            
            print("Answer Candidate: {}".format(best_candidate))
            print("Actual Answer: {}".format(answer))
            print()

if __name__ == '__main__':
    
#    my_parser = argparse.ArgumentParser()
#    my_parser.add_argument('--process_data', dest='process', default=True, action='store_false')
#    args = my_parser.parse_args()
    input_file = sys.argv[1]
    nlp = spacy.load("en_core_web_sm")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    
    #open file for processing
    story_dict, qa_dict = process_data(input_file, nlp, tokenizer, bert_model, process=False)
    #print(story_dict, qa_dict)
    create_answers(story_dict, qa_dict, nlp)







    
