# Question Answering System
This is a project created for the Fall 2022 University of Utah CS 6340 Natural Language Processing Class.  The project is to design and build a question answering system.  For the project, stories from the Canadian Broadcasting Corporation fweb page for kids were used.  MITRE Corporation created questions and an answer key for each story.  The aim of this project was to create a question answering system that could process a story and a list of questions and produce an answer for each question.  

The following libraries used for this:  
BERT-based model (bert-based-uncased) from the Huggingface library.\\
https://huggingface.co

spaCy library for NLP processing tasks\\
https://spacy.io

nltk library for NLP processing tasks\\
https://www.nlkt.org

nueralcoref module for coreferencing\\
https://spacy.io/universe/project/neuralcoref

The following data files are used for this project:\\
stopwords.txt\\
input.txt\\

The first line of the input file is a directory path.  Each subsequent line in the file is a StoryID.  For each StoryID, the directory containts a story file named StoryID.story (e.g. "1999-W02-5.story") and a question file named StoryID questions (e.g. "1999-W02-5.questions").  The QA system produces an answer for each question in the question file based on the corresponding story file.

Each story file includes a headline, date, and StoryID line followed by the text of the story.  Each question file contains 3 lines for each question indicating the QuestionID, the question itself, and a difficulty rating.

The QA system produces a single response file, printed to standard output, which contains answers for all of the stories and questions in the input file.

To run the program type the following on the command line:\\
$ python3 qa.py <input_file.txt>

To score the program, type the following on the command line:\\
scoring


