# NLP with Sequence Models

## Table of contents

- [Table of contents](#table-of-contents)
- [About this course](#about-this-course)
- [Neural Networks for Sentiment Analysis](#neural-networks-for-sentiment-analysis)
- [Trax: Neural Networks](#trax-neural-networks)
- [Reading: (Optional) Trax and JAX, docs and code](#reading-optional-trax-and-jax-docs-and-code)
- [Classes, subclasses and inheritance](#classes-subclasses-and-inheritance)
- [Dense and ReLU layer](#dense-and-relu-layer)
- [Serial Layer](#serial-layer)
- [Other Layers](#other-layers)
- [Training](#training)

## About this course

[back to TOC](#table-of-contents)

In Course 3 of the [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing), you will:

1. Train a neural network with GLoVe word embeddings to perform sentiment analysis of tweets,
2. Generate synthetic Shakespeare text using a Gated Recurrent Unit (GRU) language model,
3. Train a recurrent neural network to perform named entity recognition (NER) using LSTMs with linear layers, and
4. Use so-called ‘Siamese’ LSTM models to compare questions in a corpus and identify those that are worded differently but have the same meaning.

By the end of this Specialization, you will have designed NLP applications that perform question-answering and sentiment analysis, created tools to translate languages and summarize text, and even built a chatbot!

This Specialization is designed and taught by two experts in NLP, machine learning, and deep learning. Younes Bensouda Mourri is an Instructor of AI at Stanford University who also helped build the Deep Learning Specialization. Łukasz Kaiser is a Staff Research Scientist at Google Brain and the co-author of Tensorflow, the Tensor2Tensor and Trax libraries, and the Transformer paper.

## Neural Networks for Sentiment Analysis

[back to TOC](#table-of-contents)

Previously in the course you did sentiment analysis with logistic regression and naive Bayes. Those models were in a sense more naive, and are not able to catch the sentiment off a tweet like: "I am not happy " or "If only it was a good day". When using a neural network to predict the sentiment of a sentence, you can use the following. Note that the image below has three outputs, in this case you might want to predict, "positive", "neutral ", or "negative".

![Alt text](Images/C3W1N1_01.png)

Note that the network above has three layers. To go from one layer to another you can use a WW matrix to propagate to the next layer. Hence, we call this concept of going from the input until the final layer, forward propagation. To represent a tweet, you can use the following:

![Alt text](Images/C3W1N1_02.png)

Note, that we add zeros for padding to match the size of the longest tweet.

## Trax: Neural Networks

[back to TOC](#table-of-contents)

**Trax** has several advantages:

- Runs fast on CPUs, GPUs and TPUs
- Parallel computing
- Record algebraic computations for gradient evaluation

Here is an example of how you can code a neural network in Trax:

![Alt text](Images/C3W1N2_01.png)

## Reading: (Optional) Trax and JAX, docs and code

[back to TOC](#table-of-contents)

Official Trax documentation maintained by the Google Brain team:

<https://trax-ml.readthedocs.io/en/latest/>

Trax source code on GitHub:

<https://github.com/google/trax>

JAX library:

<https://jax.readthedocs.io/en/latest/index.html>

## Classes, subclasses and inheritance

[back to TOC](#table-of-contents)

Trax makes use of classes. If you are not familiar with classes in python, don't worry about it, here is an example.

![Alt text](Images/C3W1N3_01.png)

In the example above, you can see that a class  takes in an **init** and a **call** method. These methods allow you to initialize your internal variables and allow you to execute your function when called. To the right you can see how you can initialize your class. When you call MyClass(7) , you are setting the y variable to 7. Now when you call f(3) you are adding 7 + 3. You can change the my_method function to do whatever you want, and you can have as many methods as you want in a class.  

## Dense and ReLU layer

[back to TOC](#table-of-contents)

The Dense layer is the computation of the inner product between a set of trainable weights (weight matrix) and an input vector.  The visualization of the dense layer could be seen in the image below.

![Alt text](Images/C3W1N4_01.png)

The orange box shows the dense layer. An activation layer is the set of blue nodes. Concretely one of the most commonly used activation layers is the rectified linear unit (ReLU).

![Alt text](Images/C3W1N4_02.png)

ReLU(x) is defined as max(0,x) for any input x.

## Serial Layer

[back to TOC](#table-of-contents)

A serial layer allows you to compose layers in a serial arrangement:

![Alt text](Images/C3W1N5_01.png)

It is a composition of sublayers. These layers are usually dense layers followed by activation layers.

## Other Layers

[back to TOC](#table-of-contents)

Other layers could include embedding layers and mean layers. For example, you can learn word embeddings for each word in your vocabulary as follows:
![Alt text](Images/C3W1N6_01.png)

The mean layer allows you to take the average of the embeddings. You can visualize it as follows:
![Alt text](Images/C3W1N6_02.png)

This layer does not have any trainable parameters.

## Training

[back to TOC](#table-of-contents)

In Trax, the function grad allows you to compute the gradient. You can use it as follows:
![Alt text](Images/C3W1N7_01.png)

Now if you were to evaluate grad_f at a certain value, namely z, it would be the same as computing 6z+1.  Now to do the training, it becomes very simple:
![Alt text](Images/C3W1N7_02.png)

You simply compute the gradients by feeding in y.forward (the latest value of y), the weights, and the input x, and then it does the back-propagation for you in a single line. You can then have the loop that allows you to update the weights (i.e. gradient descent!).
