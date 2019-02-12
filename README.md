# Mini-project2

**Mini-project2_no_extra.py:**
* *Task: You must run experiments using at least two different classifiers from the SciKit learn package*
* Preprocessing: Removes punctuation, replaces special characters with spaces and get the steem of the words
* Features: Unigrams
* Validation Pipeline: Held-out validation
* Classifiers: Logistic Regression and Linear SVM

**LogReg_tfidf_pipeline.py**
* *Code to find the best set of hyperparameters for LogisticRegression using GridSearchCV*
* Preprocessing: Removes punctuation, replaces special characters with spaces and get the steem of the words
* Features: Tf-idf (unigrams-bigrams) weighting
* Validation Pipeline: Cross-validation
* Parameters_grid: vect_binary, vect_min_df, tfidf_use, classifier_penalty, classifier_C

**LogReg_length+tfidf_pipeline.py**
* *Variation from LogReg_tfidf_pipeline.py
* Adds length feature to the final pipeline

**LogReg_vect+tfidf_pipeline.py**
* *Variation from LogReg_tfidf_pipeline.py
* Adds vectorization (unigrams) features to the final pipeline

**testing_best_Logistic.py**
* *Uses the best hyperparameters found using LogisticRegression.py and executes the classifier once*

**submit_best_Log.py**
* *Reads train and test data. Uses the best model of LogRegression and outputs a csv file containing the id and category numbers*
* Best model: Compile and normalization, no stemming/lemmatization, no stop-words, uni and bigrams with tf-idf weighting, binary=True, C=100, penalty=l2
