# Class Recommendation System-Trademarkia

// AI Engineer Task

// class recommendation system 
objective
The primary objective of this task is to create an AI Model that can accurately recommend trademark classes based on goods and services entered by user, even for items that are not present in the uspto id manual.

Task
1. Develop an AI model using technique of your choice to recommend trademark classes for goods and services entered by the user. the AI model will be trained using a large dataset of existing goods and services from the USPTO ID manual.
2. Test the AI model with a wide range of inputs , including those that are not present in the USPTO ID manual.
3. Make Rest API which can be integrated by other developers to input description of thier goods and services and display the recommended class name as response.

Description of the Project:

The main goal of this project is to create an AI model that can accurately suggest trademark classes based on the descriptions of goods and services provided by users. I have trained the model using a large dataset of existing goods and services from the USPTO ID Manual, ensuring that it can provide reliable recommendations even for items that are not listed in the manual.

To achieve this, I have developed a recommendation system that combines natural language processing (NLP) techniques and deep learning models. When a user enters a description, the system preprocesses the text by breaking it down into meaningful units and adjusting the length. The preprocessed description is then fed into a trained neural network model.

During the training phase, I had experimented it with different models and techniques to optimize the accuracy of the recommendations. After thorough testing and fine-tuning, I achieved an accuracy of approximately 72%, which is a significant improvement compared to randomly guessing the trademark classes.
The system utilizes a combination of techniques such as word embedding, convolutional neural networks (CNN), and recurrent neural networks (RNN) like LSTM. These techniques allow the model to analyze the text data, identify important patterns, and understand the dependencies between words, enabling it to make informed predictions.

and also I have created a REST API that can be easily integrated by other developers. This API allows developers to input descriptions of their goods and services and receive the recommended trademark class as a response. The API is designed to be user-friendly and can handle a wide range of inputs, even descriptions that are not found in the USPTO ID Manual.

To interact with the recommendation system, I have developed a web-based interface. Users can visit the interface and find a simple form where they can enter their description and submit it for classification. The recommended trademark class is then displayed on the webpage, providing users with immediate feedback.
It's important to note that achieving 100% accuracy in trademark class recommendations is challenging due to the subjective nature of class assignments. However, My model's accuracy of 72% indicates its effectiveness in capturing patterns and providing reliable recommendations.

I was continuously working on improving the accuracy of the model by exploring advanced techniques, incorporating additional data sources, and leveraging user feedback to enhance the recommendation system.

Overall, the aim of this project is to offer a powerful and user-friendly solution for trademark class recommendation. It assists businesses and developers in accurately classifying their goods and services for intellectual property purposes, saving them time and effort in the trademark registration process.


structure for the project:

- project_directory( Any name for the folder )
  - app.py  
  - templates
    - index.html
  - mymodel.h5 (weights file)
  - tokenizer.json
  - label_encoder.json 
  - shuffled_file.json (cleaned data used for generating the weights file)
  - main.py ( "to generate the weights file i.e."mymodel.h5" )


