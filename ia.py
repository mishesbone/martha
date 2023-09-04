#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customizable Agent martha that Responds to User

Created on Thu Aug 24 21:20:55 2023

@author: roboteknologies
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import spacy
import random

class IntentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            # Convert text to TF-IDF features
            ('tfidf', TfidfVectorizer()),  
            # Linear SVM classifier
            ('clf', LinearSVC())  
        ])

    def train(self, training_data):
        intents, texts = zip(*training_data)
        self.pipeline.fit(texts, intents)

    def predict_intent(self, user_input):
        predicted_intent = self.pipeline.predict([user_input])
        return predicted_intent[0]



class IntelligentAgent:
    def __init__(self, possessor_name):
        self.possessor_name = possessor_name
        self.nlp = spacy.load("en_core_web_sm")
        self.behaviors = {}

    def customize_behavior(self, behavior, response):
        self.behaviors[behavior] = response

    def train_intent_classifier(self, training_data):
        intents = [data[0] for data in training_data]
        texts = [data[1] for data in training_data]

        for intent, text in zip(intents, texts):
            if intent not in self.behaviors:
                # Initialize as an empty list
                self.behaviors[intent] = []  
                # Append text to the list
                self.behaviors[intent].append(text) 


    def respond(self, user_input, predicted_intent):
        if predicted_intent in self.behaviors:
            possible_responses = self.behaviors[predicted_intent]
            return possible_responses[random.randint(0, len(possible_responses) - 1)]
        return "I'm not sure how to respond to that."



# Training the intent classifier with labeled training data
training_data = [
    ("greeting", "Hi there!"),
    ("greeting", "Hello!"),
    ("favorite color", "What's your favorite color?"),
    ("favorite color", "Tell me your favorite color."),
    ("greeting", "Hey!"),
    ("greeting", "Good morning!"),
    ("favorite color", "Do you have a preferred color?"),
    ("favorite color", "I'd like to know your favorite color."),
    ("farewell", "Goodbye!"),
    ("farewell", "See you later!"),
    ("farewell", "Take care!"),
    ("weather", "What's the weather like today?"),
    ("weather", "Can you tell me the current weather?"),
    ("music", "Play some music for me."),
    ("music", "I want to listen to music."),
    ("music", "What's your favorite song?"),
    ("weather", "Is it going to rain tomorrow?"),
    ("weather", "Tell me the weather forecast."),
    ("farewell", "Until next time!"),
    ("farewell", "Catch you later!"),
        ("translation", "Translate 'hello' to Spanish."),
    ("translation", "How do you say 'thank you' in French?"),
    ("recommendation", "Can you recommend a good book to read?"),
    ("recommendation", "I need a movie recommendation."),
    ("history", "Tell me about the American Civil War."),
    ("history", "What happened during the Renaissance period?"),
    ("math", "What's the square root of 144?"),
    ("math", "Calculate 25 multiplied by 3."),
    ("appointment", "Schedule a doctor's appointment for next week."),
    ("appointment", "I'd like to make a reservation at a restaurant."),
    ("movie", "What's the latest Marvel movie?"),
    ("movie", "Tell me about the film 'Inception'."),
    ("sports", "Who won the Super Bowl last year?"),
    ("sports", "What's the score of the current NBA game?"),
    ("travel", "Give me travel tips for visiting Paris."),
    ("travel", "Tell me about popular tourist destinations in Thailand."),
    ("feedback", "I have some suggestions for your website."),
    ("feedback", "I'd like to provide feedback on your customer service."),
    ("complaint", "I received a damaged product."),
    ("complaint", "Your delivery service is very slow."),
    ("location", "Where can I find a nearby restaurant?"),
    ("location", "Tell me the nearest gas station."),
    ("joke", "Tell me a joke."),
    ("joke", "Make me laugh!"),
    ("news", "What's the latest news today?"),
    ("news", "Can you update me on current events?"),
    ("reminder", "Set a reminder for my meeting at 3 PM."),
    ("reminder", "Remind me to buy groceries tomorrow."),
    ("help", "I need assistance with something."),
    ("help", "Can you help me with a problem?"),
    ("feedback", "I have some feedback to share."),
    ("feedback", "I'd like to provide feedback on your service."),
    ("appointment", "Schedule an appointment for a haircut."),
    ("appointment", "I want to book a dentist appointment."),
    ("compliment", "You're doing a great job!"),
    ("compliment", "I'm impressed with your service."),
    ("complaint", "I have a complaint about your product."),
    ("complaint", "I'm not satisfied with your service."),
]


# Create an instance of the IntentClassifier and train it with training data
intent_classifier = IntentClassifier()
intent_classifier.train(training_data)

# Create an instance of the IntelligentAgent
possessor_name = "Mishes"
agent = IntelligentAgent(possessor_name)

# Customize behaviors based on the possessor
agent.customize_behavior("greeting", f"Hello, I'm {possessor_name}'s intelligent agent!")
agent.customize_behavior("favorite color", f"{possessor_name}'s favorite color is blue.")

# Training the intent classifier with labeled training data
training_data = [
    ("greeting", "Hi there!"),
    ("greeting", "Hello!"),
    ("favorite color", "What's your favorite color?"),
    ("favorite color", "Tell me your favorite color."),
    ("greeting", "Hey!"),
    ("greeting", "Good morning!"),
    ("favorite color", "Do you have a preferred color?"),
    ("favorite color", "I'd like to know your favorite color."),
    ("farewell", "Goodbye!"),
    ("farewell", "See you later!"),
    ("farewell", "Take care!"),
    ("weather", "What's the weather like today?"),
    ("weather", "Can you tell me the current weather?"),
    ("music", "Play some music for me."),
    ("music", "I want to listen to music."),
    ("music", "What's your favorite song?"),
    ("weather", "Is it going to rain tomorrow?"),
    ("weather", "Tell me the weather forecast."),
    ("farewell", "Until next time!"),
    ("farewell", "Catch you later!"),
        ("translation", "Translate 'hello' to Spanish."),
    ("translation", "How do you say 'thank you' in French?"),
    ("recommendation", "Can you recommend a good book to read?"),
    ("recommendation", "I need a movie recommendation."),
    ("history", "Tell me about the American Civil War."),
    ("history", "What happened during the Renaissance period?"),
    ("math", "What's the square root of 144?"),
    ("math", "Calculate 25 multiplied by 3."),
    ("appointment", "Schedule a doctor's appointment for next week."),
    ("appointment", "I'd like to make a reservation at a restaurant."),
    ("movie", "What's the latest Marvel movie?"),
    ("movie", "Tell me about the film 'Inception'."),
    ("sports", "Who won the Super Bowl last year?"),
    ("sports", "What's the score of the current NBA game?"),
    ("travel", "Give me travel tips for visiting Paris."),
    ("travel", "Tell me about popular tourist destinations in Thailand."),
    ("feedback", "I have some suggestions for your website."),
    ("feedback", "I'd like to provide feedback on your customer service."),
    ("complaint", "I received a damaged product."),
    ("complaint", "Your delivery service is very slow."),
    ("location", "Where can I find a nearby restaurant?"),
    ("location", "Tell me the nearest gas station."),
    ("joke", "Tell me a joke."),
    ("joke", "Make me laugh!"),
    ("news", "What's the latest news today?"),
    ("news", "Can you update me on current events?"),
    ("reminder", "Set a reminder for my meeting at 3 PM."),
    ("reminder", "Remind me to buy groceries tomorrow."),
    ("help", "I need assistance with something."),
    ("help", "Can you help me with a problem?"),
    ("feedback", "I have some feedback to share."),
    ("feedback", "I'd like to provide feedback on your service."),
    ("appointment", "Schedule an appointment for a haircut."),
    ("appointment", "I want to book a dentist appointment."),
    ("compliment", "You're doing a great job!"),
    ("compliment", "I'm impressed with your service."),
    ("complaint", "I have a complaint about your product."),
    ("complaint", "I'm not satisfied with your service."),
]


agent.train_intent_classifier(training_data)

# Interact with the agent
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Agent: Goodbye!")
        break
    predicted_intent = intent_classifier.predict_intent(user_input)
    response = agent.respond(user_input, predicted_intent)
    print("Agent:", response)
