# NLP with Attention Models

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
- [Traditional Language models](#traditional-language-models)
- [Recurrent Neural Networks](#recurrent-neural-networks)
- [Application of RNNs](#application-of-rnns)
- [Math in Simple RNNs](#math-in-simple-rnns)
- [Cost Function for RNNs](#cost-function-for-rnns)
- [Implementation Note](#implementation-note)
- [Gated Recurrent Units](#gated-recurrent-units)
- [Deep and Bi-directional RNNs](#deep-and-bi-directional-rnns)

## About this course

[back to TOC](#table-of-contents)

In Course 3 of the [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing), you will:

1. Train a neural network with GLoVe word embeddings to perform sentiment analysis of tweets,
2. Generate synthetic Shakespeare text using a Gated Recurrent Unit (GRU) language model,
3. Train a recurrent neural network to perform named entity recognition (NER) using LSTMs with linear layers, and
4. Use so-called ‘Siamese’ LSTM models to compare questions in a corpus and identify those that are worded differently but have the same meaning.

By the end of this Specialization, you will have designed NLP applications that perform question-answering and sentiment analysis, created tools to translate languages and summarize text, and even built a chatbot!

This Specialization is designed and taught by two experts in NLP, machine learning, and deep learning. Younes Bensouda Mourri is an Instructor of AI at Stanford University who also helped build the Deep Learning Specialization. Łukasz Kaiser is a Staff Research Scientist at Google Brain and the co-author of Tensorflow, the Tensor2Tensor and Trax libraries, and the Transformer paper.

## References

[back to TOC](#table-of-contents)

This course drew from the following resources:
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al, 2019)
https://arxiv.org/abs/1910.10683

- Reformer: The Efficient Transformer (Kitaev et al, 2020)
https://arxiv.org/abs/2001.04451

- Attention Is All You Need (Vaswani et al, 2017)
https://arxiv.org/abs/1706.03762

- Deep contextualized word representations (Peters et al, 2018)
https://arxiv.org/pdf/1802.05365.pdf

- The Illustrated Transformer (Alammar, 2018)
http://jalammar.github.io/illustrated-transformer/

- The Illustrated GPT-2 (Visualizing Transformer Language Models) (Alammar, 2019)
http://jalammar.github.io/illustrated-gpt2/

- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al, 2018)
https://arxiv.org/abs/1810.04805

- How GPT3 Works - Visualizations and Animations (Alammar, 2020)
http://jalammar.github.io/how-gpt3-works-visualizations-animations/