#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
            ('tfidf', TfidfVectorizer()),  # Convert text to TF-IDF features
            ('clf', LinearSVC())  # Linear SVM classifier
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
                self.behaviors[intent] = []  # Initialize as an empty list
                self.behaviors[intent].append(text)  # Append text to the list


    def respond(self, user_input, predicted_intent):
        if predicted_intent in self.behaviors:
            possible_responses = self.behaviors[predicted_intent]
            return possible_responses[random.randint(0, len(possible_responses) - 1)]
        return "I'm not sure how to respond to that."



# Train the intent classifier with some training data
training_data = [
    ("greeting", "Hi there!"),
    ("greeting", "Hello!"),
    ("favorite color", "What's your favorite color?"),
    ("favorite color", "Tell me your favorite color.")
]
# ... (import statements and class definitions)

# Create an instance of the IntentClassifier and train it with training data
intent_classifier = IntentClassifier()
intent_classifier.train(training_data)

# Create an instance of the IntelligentAgent
possessor_name = "Mishes"
agent = IntelligentAgent(possessor_name)

# Customize behaviors based on the possessor
agent.customize_behavior("greeting", f"Hello, I'm {possessor_name}'s intelligent agent!")
agent.customize_behavior("favorite color", f"{possessor_name}'s favorite color is blue.")

# Train the intent classifier with some training data using the IntelligentAgent
training_data = [
    ("greeting", "Hi there!"),
    ("greeting", "Hello!"),
    ("favorite color", "What's your favorite color?"),
    ("favorite color", "Tell me your favorite color.")
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
