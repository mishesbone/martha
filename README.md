# martha
A Conversational AI Bot in memory of my Mum who died 14th April, 2023.

Customizable Agent "Martha" - Responds to User prompts

This Python code defines an intelligent agent named "Martha" that responds to user input. Martha is highly customizable, allowing you to train her to recognize different intents and customize her responses. This README provides an overview of how to use Martha and train her with your own data.
Table of Contents

    Introduction
    Requirements
    Usage
    Customizing Behavior
    Training the Intent Classifier
    Interacting with Martha

Introduction

Martha is an intelligent agent that uses natural language processing (NLP) techniques to classify user intents and provide responses based on those intents. The code includes two main classes:

    IntentClassifier: This class uses a linear support vector machine (LinearSVC) for intent classification. It is trained on labeled data to recognize user intents.
    IntelligentAgent: Martha, the intelligent agent, is an instance of this class. You can customize her behavior by associating different responses with specific intents.

Requirements

    Python 3.x
    Required Python packages (install them using pip):
        scikit-learn
        spacy

Usage

    Clone this repository to your local machine:

    bash

git clone https://github.com/yourusername/martha-agent.git
cd martha-agent

Run the agent:

bash

    python martha.py

    Start interacting with Martha. Type your messages, and Martha will respond based on her trained behavior.

Customizing Behavior

You can customize Martha's behavior by associating specific responses with different intents. To do this, follow these steps:

    In the agent.customize_behavior method, specify the intent and the response you want Martha to provide for that intent. For example:

    python

    agent.customize_behavior("greeting", "Hello, how can I assist you today?")
    agent.customize_behavior("farewell", "Goodbye! Have a great day!")

    These customizations will be used when Martha encounters the specified intents in user input.

Training the Intent Classifier

Martha's ability to recognize intents relies on training data. You can add your own training data by modifying the training_data list in the code. Each training example consists of an intent and a text associated with that intent. For example:

python

training_data = [
    ("greeting", "Hello there!"),
    ("farewell", "Goodbye!"),
    # Add more training examples here
]

Ensure that you have a diverse set of training examples to improve Martha's intent recognition.
Interacting with Martha

    Run the code, as mentioned in the Usage section.
    Type your messages when prompted by the agent.
    To exit the conversation, simply type "exit," and Martha will say goodbye.

Feel free to customize Martha's behavior and expand her capabilities by adding more training data and intents.

Happy conversing with Martha! 😄
