# NLP with Classification and Vector Spaces

## Table of contents

- [Table of contents](#table-of-contents)
- [Course Summary](#course-summary)
- [Vector Space Models](#vector-space-models)
- [Word by Word and Word by Doc](#word-by-word-and-word-by-doc)
  - [Word by Word Design](#word-by-word-design)
  - [Word by Document Design](#word-by-document-design)
- [LAB - Linear algebra in Python with Numpy](#lab---linear-algebra-in-python-with-numpy)
- [Euclidian Distance](#euclidian-distance)
- [Cosine Similarity: Intuition](#cosine-similarity-intuition)
- [Cosine Similarity](#cosine-similarity)
- [Manipulating Words in Vector Spaces](#manipulating-words-in-vector-spaces)
- [LAB - Manipulating word embeddings](#lab---manipulating-word-embeddings)
- [Visualization and PCA](#visualization-and-pca)
- [PCA algorithm](#pca-algorithm)
- [LAB - Another explanation about PCA](#lab---another-explanation-about-pca)

## Course Summary

## Vector Space Models

[back to TOC](#table-of-contents)

Vector spaces are fundamental in many applications in NLP. If you were to represent a word, document, tweet, or any form of text, you will probably be encoding it as a vector. These vectors are important in tasks like information extraction, machine translation, and chatbots. Vector spaces could also be used to help you identify relationships between words as follows:

![Alt text](images/C1W3N1_01.png)

The famous quote by Firth says, **"You shall know a word by the company it keeps"**. When learning these vectors, you usually make use of the neighboring words to extract meaning and information about the center word. If you were to cluster these vectors together, as you will see later in this specialization, you will see that adjectives, nouns, verbs, etc. tend to be near one another. Another cool fact, is that synonyms and antonyms are also very close to one another. This is because you can easily interchange them in a sentence and they tend to have similar neighboring words!

## Word by Word and Word by Doc

[back to TOC](#table-of-contents)

### Word by Word Design

We will start by exploring the word by word design. Assume that you are trying to come up with a vector that will represent a certain word.  One possible design would be to create a matrix where each row and column corresponds to a word in your vocabulary. Then you can iterate over a document and see the number of times each word shows up next each other word. You can keep track of the number in the matrix. In the video I spoke about a parameter KK. You can think of KK as the bandwidth that decides whether two words are next to each other or not.
![Alt text](images/C1W3N2_01.png)

In the example above, you can see how we are keeping track of the number of times words occur together within a certain distance kk. At the end, you can represent the word data, as a vector $v = [2,1,1,0]$.

### Word by Document Design

You can now apply the same concept and map words to documents. The rows could correspond to words and the columns to documents. The numbers in the matrix correspond to the number of times each word showed up in the document.
![Alt text](images/C1W3N2_02.png)

You can represent the entertainment category, as a vector $v = [500, 7000]$. You can then also compare categories as follows by doing a simple plot.
![Alt text](images/C1W3N2_03.png)

Later this week, you will see how you can use the angle between two vectors to measure similarity.

## LAB - Linear algebra in Python with Numpy

[back to TOC](#table-of-contents)

Please go through this [lecture notebook](Labs/Week%203/C1_W3_lecture_nb_01_linear_algebra.ipynb) to practice about the basic linear algebra concepts in Python using a very powerful library called **Numpy**. This will help prepare you for the graded assignment at the end of this week.

## Euclidian Distance

[back to TOC](#table-of-contents)

Let us assume that you want to compute the distance between two points: $A, B$. To do so, you can use the euclidean distance defined as

$$
d(B,A) = \sqrt{(B_1 − A_1)^2 +(B_2 − A_2)^2}
$$

​![Alt text](images/C1W3N3_01.png)

You can generalize finding the distance between the two points $(A,B)$ to the distance between an nn dimensional vector as follows:

$$
​d(\vec{v}, \vec{w}) = \sqrt{\sum_{i=1}^{n} (v_i - w_i)^2}
$$

Here is an example where I calculate the distance between 2 vectors $(n=3)$.

![Alt text](images/C1W3N3_02.png)

## Cosine Similarity: Intuition

[back to TOC](#table-of-contents)

One of the issues with euclidean distance is that it is not always accurate and sometimes we are not looking for that type of similarity metric. For example, when comparing large documents to smaller ones with euclidean distance one could get an inaccurate result. Look at the diagram below:
![Alt text](images/C1W3N4_01.png)

Normally the **food** corpus and the **agriculture** corpus are more similar because they have the same proportion of words. However the food corpus is much smaller than the agriculture corpus. To further clarify, although the history corpus and the agriculture corpus are different, they have a smaller euclidean distance. Hence $d_2 < d_1$

To solve this problem, we look at the cosine between the vectors. This allows us to compare $\beta$ and $\alpha$.

## Cosine Similarity

[back to TOC](#table-of-contents)

Before getting into the cosine similarity function remember that the norm of a vector is defined as:

$$
\| \vec{v} \| = \sqrt{\sum_{i=1}^{n} |v_i|^2 }
$$

The **dot product** is then defined as:
$$
\vec{v} \cdot \vec{w} = \sum_{i=1}^{n} v_i \cdot w_i
$$
​
![Alt text](images/C1W3N5_01.png)

The following cosine similarity equation makes sense:
$$
\cos (\beta) = \frac{\hat v \cdot \hat w}{\| \hat v \| \| \hat w \|}
$$

If $\hat v$ and $\hat w$ are the same then you get the numerator to be equal to the denominator. Hence $\beta = 0$. On the other hand, the dot product of two orthogonal (perpendicular) vectors is $0$. That takes place when $\beta = 90$.
![Alt text](images/C1W3N5_02.png)

## Manipulating Words in Vector Spaces

[back to TOC](#table-of-contents)

You can use word vectors to actually extract patterns and identify certain structures in your text. For example:
![Alt text](images/C1W3N6_01.png)

You can use the word vector for Russia, USA, and DC to actually compute a **vector** that would be very similar to that of Moscow. You can then use cosine similarity of the **vector** with all the other word vectors you have and you can see that the vector of Moscow is the closest.
![Alt text](images/C1W3N6_02.png)

Note that the distance (and direction) between a country and its capital is relatively the same. Hence manipulating word vectors allows you identify patterns in the text.

## LAB - Manipulating word embeddings

[back to TOC](#table-of-contents)

Please go through this [lecture notebook](Labs/Week%203/C1_W3_lecture_nb_02_manipulating_word_embeddings.ipynb) to apply the linear algebra concepts for the manipulation of word embeddings. This will help prepare you for the graded assignment at the end of this week.

## Visualization and PCA

[back to TOC](#table-of-contents)

Principal component analysis is an unsupervised learning algorithm which can be used to reduce the dimension of your data. As a result, it allows you to visualize your data. It tries to combine variances across features. Here is a concrete example of PCA:
![Alt text](images/C1W3N7_01.png)

Note that when doing PCA on this data, you will see that oil & gas are close to one another and town & city are also close to one another. To plot the data you can use PCA to go from $d>2$ dimensions to $d=2$.
![Alt text](images/C1W3N7_02.png)

Those are the results of plotting a couple of vectors in two dimensions. Note that words with similar part of speech (POS) tags are next to one another. This is because many of the training algorithms learn words by identifying the neighboring words. Thus, words with similar POS tags tend to be found in similar locations. An interesting insight is that synonyms and antonyms tend to be found next to each other in the plot. Why is that the case?

## PCA algorithm

[back to TOC](#table-of-contents)

PCA is commonly used to reduce the dimension of your data. Intuitively the model collapses the data across principal components. You can think of the first principal component (in a 2D dataset) as the line where there is the most amount of variance. You can then collapse the data points on that line. Hence you went from 2D to 1D. You can generalize this intuition to several dimensions.
![Alt text](images/C1W3N8_01.png)

**Eigenvector**: the resulting vectors, also known as the uncorrelated features of your data

**Eigenvalue**: the amount of information retained by each new feature. You can think of it as the variance in the eigenvector.

Also each eigenvalue has a corresponding **eigenvector**. The eigenvalue tells you how much variance there is in the eigenvector. Here are the steps required to compute PCA:
![Alt text](images/C1W4N8_02.png)

**Steps to Compute PCA:**

- Mean normalize your data
- Compute the covariance matrix
- Compute SVD on your covariance matrix. This returns $[U S V] = svd[\Sigma]$. The three matrices U, S, V are drawn above. U is labelled with eigenvectors, and S is labelled with eigenvalues.

You can then use the first n columns of vector $U$, to get your new data by multiplying $XU[:, 0:n]$.

## LAB - Another explanation about PCA

[back to TOC](#table-of-contents)

In this [lab](Labs/Week%203/C1_W3_lecture_nb_03_pca.ipynb), we are going to view another explanation about Principal Component Analysis(PCA). PCA is a statistical technique invented in 1901 by Karl Pearson that uses orthogonal transformations to map a set of variables into a set of linearly uncorrelated variables called Principal Components.

PCA is based on the Singular Value Decomposition (SVD) of the Covariance Matrix of the original dataset. The Eigenvectors of such decomposition are used as a rotation matrix. The Eigenvectors are arranged in the rotation matrix in decreasing order according to its explained variance. This last term is related to the EigenValues of the SVD.

PCA is a potent technique with applications ranging from simple space transformation, dimensionality reduction, and mixture separation from spectral information.

Follow this lab to view another explanation for PCA. In this case, we are going to use the concept of rotation matrices applied to correlated random data, just as illustrated in the next picture.
