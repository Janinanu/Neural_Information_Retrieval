# Neural_Information_Retrieval
This project explores the potential of neural NLP methods for the purpose of document similarity detection and information retrieval.

The purpose of training classifier models in the context of an information retrieval task is the following: The architectures of the classifier models described in this project contain components that yield vectors which can be regarded as semantic representations of theinput document. It is assumed that the supervised training on the document class labels (directly as in the multiclass model or indirectly as in the document pair and triplet models)will make these document representations similar for documents that belong to the sameclass and dissimilar for documents that do not belong to the same class. By feeding indocuments into the trained model and extracting the vectors from the relevant components,one can exploit the classifiers to improve the quality of the document embeddings for the information retrieval task.

The dataset used is a database of consumer complaints that were collected by the Consumer Financial Protection Bureau of the United States of America. It is downloaded from https://catalog.data.gov/dataset/consumer-complaint-database.
The complaints were submitted by customers of different US-American banks concerning issues about a variety of financial products and services. The most import fields in the dataset are the consumer complaint itself, i.e., the text that the customer submitted to the bank, the financial product or service that is concerned, e.g. ``Student Loan'' or ``Debt Collection'', as well as the issue that the customer is facing, e.g. ``Incorrect information''.
In this project, only the product and service categories, 18 in total, are used as class labels for the complaints, omitting the information from the issue categories. All complaints in the dataset are labeled. There is a total of 391,465 documents in the dataset, all of them are in English. Figure shows the frequency distribution of the 18 classes.
For the experiments, no attempts to balance out the public data are applied, because the interest lies exactly in learning to classify frequent classes.

The dataset has a structure that is useful for a multiclass classifier. However, for some experiments, the dataset has to be restructured to obtain pairs and triplets of documents. 
The document pairs and triplets are sampled in the following way:
If label XY occurs 20 times in the dataset, 20 positive document pairs (=belonging to the same class) and 20 negative document pairs (=belonging to different classes) are created. Additionally, 20 triplets are created, where the document 1 and 2 form a positive pair, and document 1 and 3 form a negative pair.

The evaluation measure used in all experiments is the Mean Average Precision (MAP).
