import sys
import spacy
import pickle
from tqdm import tqdm

def sort_questions(input_file):
    
    qa_dict = {}
    story_dict = {}
    with open(input_file) as f:
        contents = [ line.strip() for line in f ]
    directory = contents[0]
    for i in tqdm(range(1, len(contents))):
    #for i in range(1, len(contents)):
        item = contents[i]
        for file in [".story",".answers"]:                                                                                                                 
            path = directory + item + file
            with open(path) as f:
                input_data = [ line.strip() for line in f ]
                input_data = list(filter(None, input_data))
                if file == ".story":
                    story_info = {"Text":None, "Sentences":None, "Sentence_Index":None, "Embeddings":None, "NLP":None}
                    story_id = input_data[2].replace("STORYID: ","")
                    story = " ".join(input_data[4:])
                    story_info["Text"] = story
                    story_dict[story_id] = story_info        
                if file == ".answers":
                    for item in input_data:
                        if "QuestionID" in item:
                            question_id = item.replace("QuestionID: ","")
                            item_dict = {"Question":None, "Answer":None, "Difficulty":None}  
                        elif "Question" in item:
                            question = item.replace("Question: ","")
                            item_dict["Question"] = question
                        elif "Answer" in item:
                            answer = item.replace("Answer: ","")
                            item_dict["Answer"] = answer
                        elif "Difficulty" in item:
                            difficulty = item.replace("Difficulty: ", "")
                            item_dict["Difficulty"] = difficulty
                            qa_dict[question_id] = item_dict 

#        filename = "qa_dict"
#        outfile = open(filename,'wb')
#        pickle.dump(qa_dict,outfile)
#        outfile.close()
        
#        infile = open("qa_dict",'rb')
#        qa_dict = pickle.load(infile)
#        infile.close()

    print(len(qa_dict))
    return story_dict, qa_dict

def create_answer_files(qa_dict):
    
    question_types = ["Where", "Who", "When", "What", "Why", "How"]
    counts = {"Why":0, "Who":0, "What":0, "When":0, "Where":0, "How":0, "other":0}
    for question_type in question_types:   

        with open(question_type + ".answers", "w") as f:                                                                                                                               
            for question_id, question_info in qa_dict.items():
                question = question_info["Question"]
                answer = question_info["Answer"]
                difficulty = question_info["Difficulty"]
            
                if question.startswith(question_type):
                    f.write("QuestionID: {}\n".format(question_id))
                    f.write("Question: {}\n".format(question))
                    f.write("Answer: {}\n".format(answer))
                    f.write("Difficulty: {}\n".format(difficulty))
                    f.write("\n")
                    counts[question_type] +=1

    print("Finished processing questions")
    print("Final counts and percentages:")
    for question_type in question_types:
        print("{}: {}, {:.2f}".format(question_type, counts[question_type],counts[question_type]/len(qa_dict))) 


def get_dep(phrase, doc):
    text = phrase.split()
    start_pos = 0
    cur_index = 0
    dependencies = []
    dep_counts= {"nsubj":0, "pobj":0,"dobj":0,"iobj":0}
    for token in doc:
        if cur_index == len(text):
            break
        if token.text == text[cur_index]:
            if token.dep_ in ["nsubj", "pobj", "dobj", "iobj"]:
                dep_counts[token_dep] += 1
            dependencies.append(token.dep_)
            cur_index += 1
    return dep_counts, dependencies

def analyze_questions(story_dict, qa_dict, question_type):
    
    nlp = spacy.load("en_core_web_sm")
    who_counts= {"Neither":0, "Both":0,"Question":0,"Answer":0}
    total_who_count = 0
    dep_counts= {"nsubj":0, "pobj":0,"dobj":0,"iobj":0}
    dependencies = []

    for question_id, question_info in qa_dict.items():
        question = question_info["Question"]
        answer = question_info["Answer"]
        difficulty = question_info["Difficulty"]
        story_id = question_id[:10]
        doc = nlp(story_dict[story_id]["Text"])

        #Analyze Who Questions

        if question.startswith("Where"):
            
            #Analyze the entity types found in the questions and answers
            total_who_count += 1
            in_question = False
            in_answer = False
            for ent in nlp(question).ents:
                if ent.label_ == "FAC" or ent.label_ == "GPE" or ent.label_ == "LOC":
                    in_question = True
                #print(ent.text, ent.label_)
            for ent in nlp(answer).ents:
                if ent.label_ == "FAC" or ent.label_ == "GPE" or ent.label_ == "LOC":
                    in_answer = True
            if in_question:
                if in_answer:
                    who_counts["Both"] += 1
                else:
                    who_counts["Question"] += 1
            elif in_answer:
                who_counts["Answer"] += 1
            else:
                who_counts["Neither"] += 1
            
            text = answer.split()
            cur_index = 0
            for token in doc:
                if cur_index == len(text):
                    break
                if token.text == text[cur_index]:
                    if token.dep_ in ["nsubj", "pobj", "dobj", "iobj"]:
                        dep_counts[token.dep_] += 1
                    if token.dep_ == "prep":
                        print(token.text)
                    dependencies.append(token.dep_)
                    cur_index += 1

    print(dep_counts)
    print("Both: {}, Neither: {}, Question: {}, Answer: {}".format(who_counts["Both"]/total_who_count, who_counts["Neither"]/total_who_count, 
        who_counts["Question"]/total_who_count, who_counts["Answer"]/total_who_count))
    print(dependencies)
if __name__ == '__main__':
    
    input_file = sys.argv[1]
    story_dict, qa_dict = sort_questions(input_file)
    #create_answer_files(qa_dict)
    analyze_questions(story_dict, qa_dict, "Who")






    
