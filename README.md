# Neural_Information_Retrieval
This project explores the potential of neural NLP methods for the purpose of document similarity detection and information retrieval.

The dataset used is a database of consumer complaints that were collected by the Consumer Financial Protection Bureau of the United States of America. It is downloaded from https://catalog.data.gov/dataset/consumer-complaint-database.
The complaints were submitted by customers of different US-American banks concerning issues about a variety of financial products and services. The most import fields in the dataset are the consumer complaint itself, i.e., the text that the customer submitted to the bank, the financial product or service that is concerned, e.g. "Student Loan" or "Debt Collection", as well as the issue that the customer is facing, e.g. "Incorrect information".

Given a search query, the task is to find the most similar documents in the database and rank the results by a similarity score. Special focus lies on finding the most similar complaints to queries that describe frequently occuring complaints.

The purpose of training classifier models in the context of an information retrieval task is the following: The architectures of the classifier models described in this project contain components that yield vectors which can be regarded as semantic representations of theinput document. It is assumed that the supervised training on the document class labels (directly as in the multiclass model or indirectly as in the document pair and triplet models)will make these document representations similar for documents that belong to the sameclass and dissimilar for documents that do not belong to the same class. By feeding indocuments into the trained model and extracting the vectors from the relevant components, one can exploit the classifiers to improve the quality of the document embeddings for the information retrieval task.

In this project, only the product and service categories, 18 in total, are used as class labels for the complaints, omitting the information from the issue categories. All complaints in the dataset are labeled. There is a total of 391,465 documents in the dataset, all of them are in English. Figure shows the frequency distribution of the 18 classes.

<img src="https://github.com/Janinanu/Neural_Information_Retrieval/blob/master/src/public_class_distr.png" width="400" height="350" align="middle" />

For the experiments, no attempts to balance out the public data are applied, because the interest lies exactly in learning to classify frequent classes.

The dataset has a structure that is useful for a multiclass classifier. However, for some experiments, the dataset has to be restructured to obtain pairs and triplets of documents. 
The document pairs and triplets are sampled in the following way:
If label XY occurs 20 times in the dataset, 20 positive document pairs (=belonging to the same class) and 20 negative document pairs (=belonging to different classes) are created. Additionally, 20 triplets are created, where the document 1 and 2 form a positive pair, and document 1 and 3 form a negative pair.

In summary, there are four types of document representations that can be extracted from the models and used in the information retrieval task:
- Weighted average over fine-tuned word embeddings from the model's embedding layer
- Concatenated last hidden states from the bidirectional LSTM
- Linear scores resulting from the linear layer applied onto the concatenated last hidden states
- Softmax scores resulting from the softmax function applied onto the linear scores

Note, however, that only the multiclass model produces the linear and softmax scores.

The evaluation measure used in all experiments is the Mean Average Precision (MAP). It it best if the system ranks the relevant documents in high positions, while ranking the non-relevant documents in low positions.

The multiclass model is able to boost the MAP to a rather high value of around 0.65 within little above 20 epochs. The
MAP plateaus when the loss indicates that the model is starting to overfit:

<img src="https://github.com/Janinanu/Neural_Information_Retrieval/blob/master/src/public_mucl_acc.png" width="400" height="300" align="middle" />
<img src="https://github.com/Janinanu/Neural_Information_Retrieval/blob/master/src/public_mucl_loss.png" width="400" height="300" align="middle" />
<img src="https://github.com/Janinanu/Neural_Information_Retrieval/blob/master/src/public_mucl_map.png" width="400" height="300" align="middle"/>

The experiments reveal that there is a large performance gap between the multiclass models on one hand and the pair and triplet models on the other hand. The multiclass models lead to good information retrieval results within only a few epochs. By contrast, the pair and triplet models cannot produce satisfying document representations even after well over 50 and more epochs. This is particularly inefficient because one training epoch on the pairs and triplets takes much longer than one epoch on the simple multiclass dataset.

The show that the success of neural methods in the information retrieval task comes with many conditions and restrictions. Maintaining a large database of labeled documents, choosing the right model, searching the best hyperparameters, and re-training the model on new classes takes a lot of resources in terms of time and money. Applying statistical methods can in certain cases be faster and cheaper, yet, at the expense of less satisfying search results.
