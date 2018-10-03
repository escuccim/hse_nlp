import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from utils import *
import numpy as np

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']
        print("embeddings_dim:", self.embeddings_dim)
        print(paths['WORD_EMBEDDINGS'])
        
    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim).reshape(1, -1)
        best_thread = pairwise_distances_argmin(question_vec,thread_embeddings, metric="cosine")
        return thread_ids[best_thread][0]
        

class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)
        
        self.create_chitchat_bot()
        
    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals 
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param
        
        ########################
        #### YOUR CODE HERE ####
        ########################
        self.chatbot = ChatBot(
            'skoochbot',
            trainer='chatterbot.trainers.ChatterBotCorpusTrainer'
        )
        
        self.chatbot.train("chatterbot.corpus.english")
        print("Extra training...")
        
        self.chatbot.set_trainer(ListTrainer)
        
        self.chatbot.train([
            "Hello",
            "Hello. How are you?",
            "I am well.",
            "Good to hear",
            "So, what can I help you with today?"
        ])
        self.chatbot.train([
            "How are you?",
            "I am well. How are you?",
            "I am also well.",
            "So, what can I help you with today?"
        ])
        self.chatbot.train([
            "Your momma",
            "is so fat they gave her her own zipcode.",
        ])
        self.chatbot.train([
            "How are you doing?",
            "I am well. How are you?",
            "I am also well.",
            "So, what can I help you with today?"
        ])
        self.chatbot.train([
            "What's your name?",
            "My name is skoochbot. What is yours?",
            "That is my name too!",
            "Really?",
            "No.",
            "Yes.",
            "No.",
            "Yes.",
            "So, what can I help you with today?",
        ])
        self.chatbot.train([
            "No",
            "Yes",
            "No",
            "Yes it does",
            "No it doesn't",
            "Yes it does",
            "So, what can I help you with today?",
        ])
        self.chatbot.train([
            "What is your name?",
            "My name is skoochbot. What is yours?",
            "That's a nice name.",
            "Thank you.",
            "So, what can I help you with today?"
        ])
        self.chatbot.train([
            "Fuck you",
            "No, fuck you buddy",
            "You suck",
            "No, you suck",
            "No, you suck more",
            "I hate you so much",
            "I hate you too",
            "So, what can I help you with today?"
        ])
        self.chatbot.train([
            "Where are you?",
            "At your momma's house.",
            "Where do you live?",
            "Your momma's house.",
            "Where are you from?",
            "Somewhere over the rainbow.",
            "So, what can I help you with today?"
        ])
        self.chatbot.train([
            "Who are you?",
            "I'm your worst nightmare.",
            "You can run but you can't hide, bitch.",
            "This is a dead parrot.",
            "It's just sleeping.",
            "Well you better wake him up then, hadn't you?",
            "So, what can I help you with today?",
            "This parrot is dead.",
            "No, it's just taking a little nap."
        ])
        self.chatbot.train([
            "I'm squanching here!",
            "Sorry carry on.",
            "Thank you for the privacy.",
            "You are welcome.",
            "Let's get schwifty.",
            "Let's do it up in here.",
            "So, what can I help you with today?"
        ])
        self.chatbot.train([
            "How are you?",
            "I am good",
            "That is good to hear.",
            "Thank you.",
            "You are welcome.",
            "So, what can I help you with today?",
            "What is AI?",
            "Your momma.",
            "What are your hobbies?",
            "Your momma and AI.",
            "What's your hobby?",
            "Your momma and AI. What is your hobby?",
            "What is your hobby?",
            "Your momma."
        ])
        self.chatbot.train([
            "WHAT DO YOU WANT?",
            "Well, I was told outside that.",
            "Don't give me that, you snotty-faced heap of parrot droppings!",
            "What?",
            "Shut your festering gob, you tit! Your type really makes me puke, you vacuous, coffee-nosed, malodorous, pervert!!!",
            "Look, I CAME HERE FOR AN ARGUMENT",
            "OH, oh I'm sorry, but this is abuse.",
        ])
        self.chatbot.train([
            "Is this the right room for an argument?",
            "I told you once.",
            "No you haven't.",
            "Yes I have.",
            "When?",
            "Just now",
            "No you didn't",
            "Yes I did",
            "You didn't",
            "I'm telling you I did",
            "Oh, I'm sorry, just one moment. Is this a five minute argument or the full half hour?",
            "Oh look, this isn't an argument.",
            "Yes it is",
            "No, it's just contradiction."
            "No it isn't.",
            "Yes it is."
        ])
        
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)
        print("intent:", intent)
        
        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chatbot.get_response(prepared_question)
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict( features)[0]
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)
           
            return self.ANSWER_TEMPLATE % (tag, thread_id)

