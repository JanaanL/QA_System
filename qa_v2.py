import sys
import numpy as np
import spacy
import torch
import pickle
import re
import math
import neuralcoref
import copy
#import argparse
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from nltk.stem.lancaster import LancasterStemmer
from numpy.linalg import norm

def process_stopwords():                                                                                                                                              
    stopwords = set()
    filename = "stopwords.txt"
    with open(filename) as f:
        line = f.readline()
        while line:
            if line != '':
                stopwords.add(line.lower().replace("\n",""))
                line = f.readline()
        return stopwords

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
   
    return start_pos

def keyword_distance(keywords, candidate, c_index, doc):
    #print("Candidate: {}".format(candidate))
    st = LancasterStemmer()
    keywords_dist = np.array(np.ones(len(keywords)) * np.inf)
#    print("Keywords: {}".format(keywords))
    index = 0
    for keyword, weight in keywords.items():
        min_dist = np.inf
        for token in doc:
            if st.stem(keyword) == st.stem(token.text.lower()):
            #if keyword == token.text.lower():
                dist = abs(c_index - token.i)
                if dist < min_dist:
                    min_dist = dist
        keyword_distance = min_dist / weight
        keywords_dist[index] = keyword_distance    
        index += 1

    keywords_dist[keywords_dist == np.inf] = 0
    average = np.mean(keywords_dist)
    #print("average keyword distance: {}".format(average))
    if average != 0:
        return 1.0 / average
    else:
        return 0
            

def special_score_heuristics(question_type, candidate, index, doc):
    score = 0
    candidate_words = list(candidate.split())
    end_index = index + len(candidate_words) - 1
    for token in doc:
        if token.i >= index and token.i <= end_index:
    #if question_type in ['where', 'who']:
            if token.dep_ == 'pobj' or token.dep_ == 'nsubj':
                #print(token.text)
                score = 0.10
                break
    return score

def update_candidates(question, candidates, doc, preps = None):
    
    if preps is not None:
        for token in doc:
            if token.pos_ == "ADP" and token.text in preps:
                pp = ' '.join([tok.orth_ for tok in token.subtree])
                candidates[pp] = token.i
    
    for chunk in doc.noun_chunks:
        for candidate in copy.deepcopy(candidates):
           if candidate in chunk.text:
               del candidates[candidate]
               candidates[chunk.text] = chunk.start
     #          print("updating candidate with noun chunk")

    for candidate in copy.deepcopy(candidates):
        candidate_words = candidate.split(" ")
        num_matches = 0
        for word in candidate_words:
            if word in question:
                num_matches +=1
        if num_matches / len(candidate_words) >= 0.75:
            del candidates[candidate]
      #      print("deleting candidate")
    
    return candidates

def get_coreferences(story, nlp):

    doc = nlp(story)
    if doc._.has_coref:
        resolved_doc = doc._.coref_resolved
        return resolved_doc
    else:
        return story

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
                        story_info = {"Text":None, "Sentences":None, "Sentence_Index":None, "Embeddings":None, "NLP":None}
                        story_id = input_data[2].replace("STORYID: ","")
                        story = " ".join(input_data[4:])
                        new_story = get_coreferences(story, nlp)
                        story_info["Text"] = new_story
                        doc = nlp(new_story)
                        sentences = [sent.text for sent in doc.sents]
                        story_info["Sentences"] = sentences
                        sentence_start_index = [sent.start for sent in doc.sents]
                        story_info["Sentence_Index"] = sentence_start_index
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
#
#    else:
#        infile = open("story_dict",'rb')
#        story_dict = pickle.load(infile)
#        infile.close()
#        
#        infile = open("qa_dict",'rb')
#        qa_dict = pickle.load(infile)
#        infile.close()
#
    return story_dict, qa_dict

def create_answers(story_dict, qa_dict, nlp, bert_model, tokenizer):
    st = LancasterStemmer()
    stopwords = process_stopwords()
    question_words = ["who", "what", "when", "where", "why", "how"]
    
    for question_id, question_info in qa_dict.items():
        question = question_info["Question"]
        #print("Question: {}".format(question))
        story_id = question_id[:10]
        question_embedding = question_info["Embedding"]
        answer = question_info["Answer"]
        candidates = {}
        keywords = {}
        #doc = nlp(story_id["Text"]
        doc = story_dict[story_id]["NLP"]
        q_tokens = nlp(question)

        possessive = False
        for token in q_tokens:
            weight = 1
            word = token.text.lower()
            if word not in stopwords and word not in question_words and token.dep_!= "punct":
                if possessive:
                    weight = 2
                    possessive = False
                elif word == "'s":
                    possessive = True
                elif token.pos_ == "VERB":
                    weight = 3
                keywords[word] = weight    
#        print("KEYWORDS: ")
#        print(keywords)
#        print()

        #WHERE questions
        if question.startswith("Where"):
            question_type = "where"
            preps = ["in", "at", "on", "into", "inside", "outside"]
            for ent in doc.ents:
#                print(ent.text, ent.label_)
                if ent.label_ in ["LOC", "FAC", "GPE"]:
                    candidates[ent.text] = ent.start
            candidates = update_candidates(question, candidates, doc, preps)
            
       #WHO questions        
        elif  question.startswith("Who"):
            question_type = "who"
            for ent in doc.ents:
                if ent.label_ in ["GPE","NORP","ORG","PERSON"]:
                    candidates[ent.text] = ent.start
            candidates = update_candidates(question, candidates, doc, preps=None)

        #WHEN questions
        elif question.startswith("When"):
            question_type = "when"
            from_to_candidates = re.search("^[Ff]+rom [\d\w\s,.]+ to [\d\w\s,.]+$", story_dict[story_id]["Text"])
            if from_to_candidates is not None:
                for candidate in from_to_candidates:
                    candidate = candidate.group()
                    index = find_position_of_phrase(candidate, doc)
                    candidates[candidate] = index 

            for token in doc:
                if token.text == "when":
                    head = token.head
                    phrase = ""
                    for word in head.subtree:
                        phrase = phrase + word.text + " "
                    candidates[phrase] = head.i

            for ent in doc.ents:
                if ent.label_ in["DATE","EVENT","TIME"]:
                    candidates[ent.text] = ent.start
            preps = ["during", "after", "before", "while", "as"]
            candidates = update_candidates(question, candidates, doc, preps)

        #HOW MANY
        elif question.startswith("How"):
            question_type = "how"
            second_word = question.split(' ')[1]
            if second_word == "many": 
                for ent in doc.ents:
                    if ent.label_ in ["CARDINAL","PERCENT","QUANTITY"]:
                        candidates[ent.text] = ent.start

            #HOW MUCH
            elif second_word == "much":
                for ent in doc.ents:
                    if ent.label_ in ["MONEY","PERCENT"]:
                        candidates[ent.text] = ent.start

            else:
                for q_token in q_tokens:
                    if q_token.text == second_word:
                        if q_token.pos_ in ["ADV", "ADJ"]:
                            for ent in doc.ents:
                                if ent.label_ in ["CARDINAL", "ORDINAL", "QUANTITY"]:
                                    candidates[ent.text] = ent.start

                        if q_token.pos_ in ["AUX", "VERB"]:
                            for word in q_tokens:
                                if word.dep_ == "nsubj":
                                    subject = word.text
 #                                   print("Subject {}".format(subject))
                                    for w in doc:
                                        if w.text == subject:
                                            subtree = w.head.subtree
                                            phrase = ""
                                            for word in subtree:
                                                if word.dep_ in ["punct", "case", "neg"]:
                                                    phrase = phrase[:-1]
                                                if word.text in ["-","$"]:
                                                    phrase = phrase + word.text
                                                else:
                                                    phrase = phrase + word.text + " "
                                            candidates[phrase] = w.i
                                            #candidates[phrase] = find_position_of_phrase(phrase, doc)
#                                            print("Phrase added: {}".format(phrase))
            candidates = update_candidates(question, candidates, doc, preps = None)
       
#        #WHY questions
#        elif question.startswith("Why"):
#            for token in doc:
#                if token.text in ["want", "because"]:
#                    head = token.head
#                    phrase = ""
#                    for word in head.subtree:
#                        phrase = phrase + word.text + " "
#                        candidates[phrase] = token.head.i
#        
#            #Add sentences before and after the sentence that most closely matches the question    
#                max_score = -math.inf
#                sentence_index = 0
#                for index, sentence in enumerate(story_dict[story_id]["Sentences"]):
#                    sentence_embedding = story_dict[story_id]["Embeddings"][index]
#                    similarity_score = np.dot(question_embedding, sentence_embedding.T)/(norm(question_embedding) * norm(sentence_embedding))
#                    if similarity_score > max_score:
#                        max_score = similarity_score
#                        sentence_index = index
#               
#                prior_sentence = story_dict[story_id]["Sentences"][index-1]
#                candidates[prior_sentence] = story_dict[story_id]["Sentence_Index"][index-1]
#                if index + 1 < len(story_dict[story_id]["Sentences"]):
#                    post_sentence = story_dict[story_id]["Sentences"][index+1]
#                    candidates[post_sentence] = story_dict[story_id]["Sentence_Index"][index+1]
       
       #if question.startswith("What"):
#        elif question.startswith("What"):
#            for word in q_tokens:
#                if word.dep_ == "nsubj":
#                    subject = word.text
#                    for w in doc:
#                        if w.text == subject:
#                            subtree = w.head.subtree
#                            phrase = ""
#                            for word in subtree:
#                                if word.dep_ in ["punct", "case", "neg"]:
#                                    phrase = phrase[:-1]
#                                if word.text in ["-","$"]:
#                                    phrase = phrase + word.text
#                                else:
#                                    phrase = phrase + word.text + " "
#                            candidates[phrase] = w.i
        
        else:
       #if question.startswith("What"):
            for word in q_tokens:
                if word.dep_ == "nsubj":
                    subject = word.text
                    for w in doc:
                        if w.text == subject:
                            subtree = w.head.subtree
                            phrase = ""
                            for word in subtree:
                                if word.dep_ in ["punct", "case", "neg"]:
                                    phrase = phrase[:-1]
                                if word.text in ["-","$"]:
                                    phrase = phrase + word.text
                                else:
                                    phrase = phrase + word.text + " "
                            candidates[phrase] = w.i
#                           print("Phrase added: {}".format(phrase))

            if len(candidates) == 0:
                max_score = -math.inf
                phrase = ""
                sentence_start_pos = 0
                for index, sentence in enumerate(story_dict[story_id]["Sentences"]):
                    sentence_embedding = story_dict[story_id]["Embeddings"][index]
                    #similarity_score = cosine_similarity(question_embedding, sentence_embedding)
                    similarity_score = np.dot(question_embedding, sentence_embedding.T)/(norm(question_embedding) * norm(sentence_embedding))
                    if similarity_score > max_score:
                        max_score = similarity_score
                        phrase = sentence
                        sentence_start_pos = story_dict[story_id]["Sentence_Index"][index]
                candidates[phrase] = sentence_start_pos

        max_score = -math.inf 
        best_candidate = ""
        similarity_score = 0
        for candidate, c_index in candidates.items():
            #print("Candidate for consideration: {}".format(candidate))
            for s_index, sentence in enumerate(story_dict[story_id]["Sentences"]):
                sentence_start_pos = story_dict[story_id]["Sentence_Index"][s_index]
                if s_index == len(story_dict[story_id]["Sentences"]) - 1:
                    sentence_end_pos = np.inf
                else:
                    sentence_end_pos = story_dict[story_id]["Sentence_Index"][s_index+1] - 1
                #print("Candidate index: {}, sentence start position: {}, sentence end position: {}".format(c_index, sentence_start_pos, sentence_end_pos))
                if c_index >= sentence_start_pos and c_index < sentence_end_pos:
                    sentence_embedding = story_dict[story_id]["Embeddings"][s_index]
                    similarity_score = np.dot(question_embedding, sentence_embedding.T)/(norm(question_embedding) * norm(sentence_embedding))
                    #similarity_score = cosine_similarity(question_embedding, sentence_embedding)[0][0]
                    
                
            if len(keywords) > 0:
                keywords_score = keyword_distance(keywords, candidate, c_index, doc)
            else:
                keywords_score = 0
      #      print("Keywords score: {}".format(keywords_score))
            additional_score = special_score_heuristics(question_type, candidate, c_index, doc)
            score = similarity_score + keywords_score + additional_score
#            print("Candidate: {}".format(candidate))
#            print("Score: {}".format(score))

            if score > max_score:
                max_score = score
                best_candidate = candidate
#        print("Question: {}".format(question))
#        print("Answer Candidate: {}".format(best_candidate))
#        print("Answer: {}".format(answer))
#        print()
        print("QuestionID: {}".format(question_id))
        print("Answer: {}".format(best_candidate))            
        print()
#        if question.startswith("Who"):
#            print("QuestionID: {}".format(question_id))            
#            print("Answer: {}".format(best_candidate))
#            print()
#            print("Question: {}".format(question))
#            print("Answer Candidate: {}".format(best_candidate))
#            print("Answer: {}".format(answer))
#            print()

if __name__ == '__main__':
    
    input_file = sys.argv[1]
    nlp = spacy.load("en_core_web_sm")
    neuralcoref.add_to_pipe(nlp)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    
    #open file for processing
    story_dict, qa_dict = process_data(input_file, nlp, tokenizer, bert_model, process=True)
    #print(story_dict, qa_dict)
    create_answers(story_dict, qa_dict, nlp, bert_model, tokenizer)







    
