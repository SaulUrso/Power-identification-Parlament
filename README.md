# NLP-project
Project for the HLT course year23/24 @Unipi 

To execute the virtual enviroment:
1 python -m venv myenv  create a virtual enviroment using venv
2 myenv\Scripts\activate to activate the enviroment in windows
  source myenv/bin/activate to activate the enviroment in macOS and Linux
3 pip install -r requirements.txt to install the requirements
4 pip list to verify the correct installation of the requirements
5 deactivate to deactivate the enviroment

To run the experiments on the random forest run "Random_Forest_Final.ipynb" with "preprocessing_utils.py" in the same folder.

TF-IDF embeddings.ipynb: In this file we create the tfidf embeddings and we try them on a logistic regression. This file will create a file with the results of the grid search over the hyperparameters of the logistic regression and a file of the results over the the test set of the logistic regression with the best combination of the hyperparameters

w2w_embeddings.ipynb: In this file we create the wor2vec embeddings and we try them on a logistic regression. In this file we apply a grid search over the possible values of the context window and vector size of word2vec. For each combination of this values a file is created and these file contain the word2vec models with that specific combination of context window and vector size. w2w_embeddings.ipynb will also create, for each word2vec model, a file with the results of the grid search over the hyperparameters of the logistic regression. After that we take the best model, basend on the performance of the logistic regression on validation set, and we create a file of the results ,over the the test set, of the logistic regression with the best combination of the hyperparameters for the best word2vec model. We also compute the word analogy for the best model

fast_text.ipynb: In this file we create the fasttext embeddings and we try them on a logistic regression. In this file we apply a grid search over the possible values of the context window and vector size of fasttext. For each combination of this values a file is created and these file contain the fasttext models with that specific combination of context window and vector size. fast_text.ipynb will also create, for each fasttext model, a file with the results of the grid search over the hyperparameters of the logistic regression. After that we take the best model, basend on the performance of the logistic regression on validation set, and we create a file of the results ,over the the test set, of the logistic regression with the best combination of the hyperparameters for the best fasttext model. We also compute the word analogy for the best model

pre trained w2v.ipynb: In this file we work with pretrained models and fine tuned models. In this file you can test differnt pretrained models, the ones avilable on gensim, over a logistic regression. In this code a file will be produce for every pretrained model that is tried on the logistic regression (we apply a grid search over the hyperparameters of the logistic regression). We also provide metods to make a fine tuning, starting from the pretrained embeddings, of word2vec models and fasttext models( we use 2 different tokenizer: punkt and bert_based_uncased). Also in this case different file will be produced with the results of the fine tuning model over the logist regression. The results provided on this file are only about the validation set because the scores are not good enough.

To execute in the correct way these 4 files(TF-IDF embeddings.ipynb, w2w_embeddings.ipynb, fast_text.ipynb,pre trained w2v.ipynb) a folder 'Result' and 'Embeddings' is needed and the file datautils.py needs to be in the same folder.

To fine-tune  the pretrained models, we provide you a kaggle notebook. Notice that the environment used there is different from the one provided in requirements.txt.
Here there is the url of the Kaggle notebook: https://www.kaggle.com/code/saulurso/finetuning-politics

requiremnts.txt: Is the file that contains the dipendences for the python enviroment
