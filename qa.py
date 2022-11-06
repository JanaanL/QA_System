print("Best candidate: {}".format(candidate))
        print("Answer: {}".format(questions["Answer"]))

#        max_score = 0
#        best_sentence = ""
#        for sentence_info in signature_vectors[story_id]:
#            signature_vector = sentence_info["Vector"]
#            score = cosine_similarity(question_vector, signature_vector)[0][0]
#            if score > max_score:
#                max_score = score
#                best_sentence = sentence_info["Text"]
#        print("Question: {}".format(questions["Question"]))
#        print("Best matching sentence: {}".format(best_sentence))
#        doc = nlp(best_sentence)
#        for ent in doc.ents:
#            print(ent.text, ent.start_char, ent.end_char, ent.label_)
#or token in doc:
            #print(token.text, token.tag_, token.dep_)
        #for ent in doc.ents:
        #    print(ent.text, ent.start_char, ent.end_char, ent.label_)
        print("\n")
    



if __name__ == '__main__':

    input_file = sys.argv[1]
    nlp = spacy.load("en_core_web_sm")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    
    #open file for processing
    input_data = get_input_data(input_file, nlp, tokenizer, bert_model)
    #unpack data
    story_dict, qa_dict, signature_vectors = input_data

    get_answers(story_dict, qa_dict, signature_vectors, nlp, tokenizer, bert_model)







    
