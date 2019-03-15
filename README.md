# Cognitive-Project
Deep learning and AI projects
The project is focussed on:-
IT Support Ticket Classification and Deployment

Project Description and initial assumptions:

As a part of our final project for Cognitive computing, we decided to address a real life business challenge for which we chose IT Service Management. Of all the business cases, we were interested with four user cases that might befitting for our project.

1. In Helpdesk, almost 30-40% of incident tickets are not routed to the right team and the tickets keep roaming around and around and by the time it reaches the right team, the issue might have widespread and reached the top management inviting a lot of trouble.
 
2. Let’s say that users are having some trouble with printers. User calls help desk, he creates a ticket with IT Support, and they realize that they need to update a configuration in user’s system and then they resolve the ticket. What if 10 other users report the same issue. We can use this data and analyze the underlying issue by using unstructured data. Also, we can analyze the trend and minimize those incident tickets.
 
3. Virtual bot that can respond and handle user tickets.
Of the three use cases, we decided to go with the first use case for now owing to our time and resource limitation.

Tools and technologies used:

● ServiceNow Instance( an IT Service management      
    platform) 
● AWS Lambda function
● EC2 Instance
● Python
● Neural Network models
● 50000+ sample tickets from open sources

Dataset:
 
We pulled the dataset directly from servicenow and tried to classify them using ML techniques to make a labelled dataset. We just kept the description and categories from the ticket and went ahead with analytics on it to come up with top 5 categories for incidents.

The labels were added once we have classified the top 5 incident categories using Topic Modelling 
 
Use Case Overview:

Most of the critical issues are handled and being tracked in the form of tickets in a typical IT environment. IT infrastructure are group of components connected together in a network. So, when a component goes down,  it takes many other components along with it. It may even take days to figure out what went wrong and find the root cause. So, it is essential to notify the component owners 
Immediately so that proper measures can be taken to prevent this from happening.

These issues are usually reported by users to helpline or Service Desk who are the first line of support. Unfortunately, it is difficult for anyone to be aware of all the components and hence tickets are being routed to the wrong team. For it to reach the original team it might even take days or weeks.Hence, we thought we could use NLP to solve this issue.

IT Service Management, on an average generates more than 40 GB of data. That gives us enough data to train and implement our model.

Our model when properly implemented could save billions of dollars to business units, as they lose many Service level agreements because of this.

Cloud-computing based ITSM:

This business case was almost impossible to handle before the arrival of cloud computing. Cloud computing made this integration between machine learning algorithms and IT Service Management possible at the first place.





For our business case, we used a cloud-based IT Service Management application named Servicenow and we accessed the Machine learning model wrapped inside the AWS lambda function using API gateway.

Integration between ServiceNow and AWS:





Topic Modelling for Categorization
 
·   	We pick the number of topics ahead of time even if we’re not sure what the topics are.
·   	Each document is represented as a distribution over topics.
·       Each topic is represented as a distribution over words.
 
We used NLTK’s Wordnet to find the meanings of words, synonyms, antonyms, and more. In addition, we use WordNetLemmatizer to get the root word.
 
We then read our dataset line by line and prepare each line for LDA and store in a list.
 
LDA with Gensim
First, we are creating a dictionary from the data, then convert to bag-of-words corpus and save the dictionary and corpus for future use.

 
 
We then tried finding out 5 topics using LDA :-
 
 
 
 
pyLDAvis
pyLDAvis is designed to help users interpret the topics in a topic model that has been fit to a corpus of text data. The package extracts information from a fitted LDA topic model to inform an interactive web-based visualization.
Visualizing 5 topics:

 
 
 
 
From Topic Modelling we came to conclusion that the whole dataset can be divided into 5 categories: -
 
·  	Network
·  	User Maintenance
·  	Database
·  	Application Workbench
·  	Security
 
We then labelled our dataset accordingly and prepared a dataset to perform supervised learning on it.
 
RNN for Classification:-
 
An end-to-end text classification pipeline is composed of following components:
1.	Training text: It is the input text through which our supervised learning model is able to learn and predict the required class.
2.   Feature Vector: A feature vector is a vector that contains information describing the characteristics of the input data.
3.   Labels: These are the predefined categories/classes that our model will predict
4.   ML Algo: It is the algorithm through which our model is able to deal with text classification (In our case : CNN, RNN, HAN)
5.   Predictive Model: A model which is trained on the historical dataset which can perform label predictions.


 
 
Incident Classification Using Recurrent Neural Network (RNN) :
 
A recurrent neural network (RNN) is a class of artificial neural network where connections between nodes form a directed graph along a sequence. This allows it to exhibit dynamic temporal behavior for a time sequence.
Using the knowledge from an external embedding can enhance the precision of your RNN because it integrates new information (lexical and semantic) about the words, an information that has been trained and distilled on a very large corpus of data.The pre-trained embedding we used is GloVe.
 
RNN is a sequence of neural network blocks that are linked to each others like a chain. Each one is passing a message to a successor.
 
 a chunk of neural network, AA, looks at some input xtxt and outputs a value htht. A loop allows information to be passed from one step of the network to the next.
These loops make recurrent neural networks seem kind of mysterious. However, if you think a bit more, it turns out that they aren’t all that different than a normal neural network. A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. Consider what happens if we unroll the loop:
 
 
LSTM Networks
Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work.1 They work tremendously well on a large variety of problems, and are now widely used.
LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!
All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.

The repeating module in a standard RNN contains a single layer.
LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

The repeating module in an LSTM contains four interacting layers.
Don’t worry about the details of what’s going on. We’ll walk through the LSTM diagram step by step later. For now, let’s just try to get comfortable with the notation we’ll be using.

In the above diagram, each line carries an entire vector, from the output of one node to the inputs of others. The pink circles represent pointwise operations, like vector addition, while the yellow boxes are learned neural network layers. Lines merging denote concatenation, while a line forking denote its content being copied and the copies going to different locations.
The Core Idea Behind LSTMs
The key to LSTMs is the cell state, the horizontal line running through the top of the diagram.
The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.

The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.
Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.

The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”
An LSTM has three of these gates, to protect and control the cell state.
Step-by-Step LSTM Walk Through
The first step in our LSTM is to decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.” It looks at ht−1ht−1and xtxt, and outputs a number between 00 and 11 for each number in the cell state Ct−1Ct−1. A 11represents “completely keep this” while a 00 represents “completely get rid of this.”
Let’s go back to our example of a language model trying to predict the next word based on all the previous ones. In such a problem, the cell state might include the gender of the present subject, so that the correct pronouns can be used. When we see a new subject, we want to forget the gender of the old subject.

The next step is to decide what new information we’re going to store in the cell state. This has two parts. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values, C̃ tC~t, that could be added to the state. In the next step, we’ll combine these two to create an update to the state.
In the example of our language model, we’d want to add the gender of the new subject to the cell state, to replace the old one we’re forgetting.

It’s now time to update the old cell state, Ct−1Ct−1, into the new cell state CtCt. The previous steps already decided what to do, we just need to actually do it.
We multiply the old state by ftft, forgetting the things we decided to forget earlier. Then we add it∗C̃ tit∗C~t. This is the new candidate values, scaled by how much we decided to update each state value.
In the case of the language model, this is where we’d actually drop the information about the old subject’s gender and add the new information, as we decided in the previous steps.

Finally, we need to decide what we’re going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through tanhtanh (to push the values to be between −1−1 and 11) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.
For the language model example, since it just saw a subject, it might want to output information relevant to a verb, in case that’s what is coming next. For example, it might output whether the subject is singular or plural, so that we know what form a verb should be conjugated into if that’s what follows next.

Variants on Long Short Term Memory
What I’ve described so far is a pretty normal LSTM. But not all LSTMs are the same as the above. In fact, it seems like almost every paper involving LSTMs uses a slightly different version. The differences are minor, but it’s worth mentioning some of them.
One popular LSTM variant, introduced by Gers & Schmidhuber (2000), is adding “peephole connections.” This means that we let the gate layers look at the cell state.

The above diagram adds peepholes to all the gates, but many papers will give some peepholes and not others.
Another variation is to use coupled forget and input gates. Instead of separately deciding what to forget and what we should add new information to, we make those decisions together. We only forget when we’re going to input something in its place. We only input new values to the state when we forget something older.

A slightly more dramatic variation on the LSTM is the Gated Recurrent Unit, or GRU, introduced by Cho, et al. (2014). It combines the forget and input gates into a single “update gate.” It also merges the cell state and hidden state, and makes some other changes. The resulting model is simpler than standard LSTM models, and has been growing increasingly popular.

 
 
Our Architecture
General Model


 
 
LSTM Model
 

To use Keras on text data, we first have to preprocess it. For this, we can use Keras’ Tokenizer class. This object takes as argument num_words which is the maximum number of words kept after tokenization based on their word frequency.
 
Once the tokenizer is fitted on the data, we can use it to convert text strings to sequences of numbers. These numbers represent the position of each word in the dictionary (think of it as mapping).
·   	In this project, we tried to tackle the problem by using recurrent neural network and attention based LSTM encoder.
·   	By using LSTM encoder, we intent to encode all the information of text in the last output of Recurrent Neural Network before running feed forward network for classification.
·       This is very similar to neural translation machine and sequence to sequence learning.
·       We used LSTM layer in Keras to address the issue of long term dependencies.
Model scoring and selection
Our model scoring and selection is based on the standard evaluation metrics Accuracy, Precision, and F1 score, which are defined as follows:

where:
·   	TP represents the number of true positive classifications. That is, the records with the actual label A that have been correctly classified, or „predicted”, as label A.
·   	TN is the number of true negative classifications. That is, the records with an actual label not equal to A that have been correctly classified as not belonging to label A.
·   	FP is the number of false positive classifications, i.e., records with an actual label other than A that have been incorrectly classified as belonging to category A.
·       FN is the number of false negatives, i.e., records with a label equal to A that have been incorrectly classified as not belonging to category A.

Figure 3. “Confusion matrix” of predicted vs. actual categorizations
The combination of predicted and actual classifications is known as „confusion matrix”, illustrated in Figure 3. Against the background of these definitions, the various evaluation metrics provide the following insights:
·   	Accuracy: The proportion of the total number of model predictions that were correct.
·   	Precision (also called positive predictive value): The proportion of correct predictions relative to all predictions for that specific class.
·   	Recall (also called true positive rate, TPR, hit rate or sensitivity): The proportion of true positives relative to all the actual positives.
·   	False positive rate (also called false alarm rate): The proportion of false positives relative to all the actual negatives (FPR).
·       F1: (The more robust) Harmonic mean of Precision and Recall.
 
Outputs:-
 General RNN Model:-

 
 
RNN
 

 
 
Observations:-
The general RNN model without LSTM provided us the accuracy of 66% whereas we were able to increase it to 69% using LSTM network layer in RNN
 
The low volume of data resulted in accuracy being almost stagnant after 70% and it didn’t matter whether we increased the epochs as was evident in the plot.
 
Yet 69% was fair enough classification as we intend to train it online and continuously improve the accuracy as volume of data grows to higher degree.
 


Model Deployment
 
For the scope of this project, we planning to deploy the model on Amazon AWS and integrate it with Service Now so that the model do the online or real-time predictions. To perform this we will first export the model by dumping it into an pickle file. Also, we will write a function which will connect to the S3 bucket and fetch and read the pickle file from there and recreate the model. 

So the workflow looks like this:

Create the incident in Service Now
The incident is received in AWS and our AWS EC2 instance or service is running
Fetch the function.py file from S3 bucket which will read the model from Pickle file and recreate the model
It will extract the feature from the service request i.e. description of the incident
Now, the code will be executed in AWS Lambda and it will provide us the category to which incident belongs


Create an EC2 Instance on AWS: -
First, Create an AWS account  which will give you free usage for 1-year on some limited services, but is enough for this project
Create an EC2 instance and select the free tier machine or if you have credits in your account and need a more powerful machine you can select from other options 

Configure a virtual runtime environment for Python on AWS and once you are done then zip all the configurations files into a single file and also include the function.py file in the zip file as we will upload this file in AWS S3 bucket
Create an S3 bucket and upload your pickle file which contains model’s details like model name, hyperparameters, and weights of the model and also uploads the zip file which contains the function.py file along with configuration settings for python virtual environment

Now, lastly setup AWS Lambda this is where we will run the python script and do the predictions



AWS Lambda is a compute service that let you run the code without any need of provisioning or managing the servers. It takes care of that part itself. The best part about AWS Lambda is that you pay only for the compute time you consume - there is no charge when your code is not running. With AWS Lambda, you can run code for virtually any type of application or backend service - all with zero administration. AWS Lambda runs your code on a high-availability compute infrastructure and performs all of the administration of the compute resources, including server and operating system maintenance, capacity provisioning and automatic scaling, code monitoring and logging. All you need to do is supply your code in one of the languages that AWS Lambda supports.


Though we were not able to configure the AWS lambda as none of us was familiar with it and have to go through the documentation and fell short of time. We are planning to extend the project for our other course and complete it. We will update the blog once we achieve our goal.






Resources:-

https://medium.com/jatana/report-on-text-classification-using-cnn-rnn-han-f0e887214d5f 
https://medium.com/datadriveninvestor/automation-all-the-way-machine-learning-for-it-service-management-9de99882a33
https://github.com/karolzak/support-tickets-classification


