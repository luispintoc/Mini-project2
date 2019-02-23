# Mini-project2
In order to run these scripts, the positive-words and negative-words textfiles should be in the same folder. Also, the packages used are scikit-learn, nltk and pattern. Moreover, run this nltk.download('movie_reviews') to download one of the lexicon.

**LinearSVC_gridsearch.py**
* *Task: Gets the k best features using GridSearchCV and SelectKBest(chi2)
* Preprocessing: Removes punctuation, replaces special characters with spaces
* Features: Uni-grams, bi-grams and tri-grams using tf-idf vectorization
* Validation Pipeline: 4-fold cross validation
* Classifiers: Linear SVC

**LogRegression_gridsearch.py**
* *Task: Gets the k best features using GridSearchCV and SelectKBest(chi2)
* Preprocessing: Removes punctuation, replaces special characters with spaces
* Features: Uni-grams and bi-grams using tf-idf vectorization
* Validation Pipeline: 4-fold cross validation
* Classifiers: Logistic Regression

**RandomForest_gridsearch.py**
* *Task: Gets the k best features using GridSearchCV and SelectKBest(chi2)
* Preprocessing: Removes punctuation, replaces special characters with spaces
* Features: Uni-grams, bi-grams and tri-grams using tf-idf vectorization
* Validation Pipeline: 4-fold cross validation
* Classifiers: Random Forest

**Log+SVC_gridsearch.py**
* *Task: Gets the k best features using GridSearchCV and SelectKBest(chi2)
* Preprocessing: Removes punctuation, replaces special characters with spaces
* Features: Uni-grams, bi-grams and tri-grams using tf-idf vectorization and also uni-grams using count vectorization and the Bing Liu's opinion lexicon
* Validation Pipeline: 4-fold cross validation
* Classifiers: Ensemble of Logistic Regression and Linear SVC

**SVC+Log+Rf_gridsearch.py**
* *Task: Gets the k best features using GridSearchCV and SelectKBest(chi2)
* Preprocessing: Removes punctuation, replaces special characters with spaces
* Features: Uni-grams, bi-grams and tri-grams using tf-idf vectorization and also uni-grams using count vectorization and the Bing Liu's opinion lexicon
* Validation Pipeline: 4-fold cross validation
* Classifiers: Ensemble of Logistic Regression, Linear SVC and Random Forest

**Standard_algorithm_printCSV.py**
* *Outputs a CSV file to submit on Kaggle evaluating the model on the test set*
* *Only works for Linear SVC, Logistic Regression and Random Forest (uncomment to test each classifier)*

**Ensemble_algorithm_printCSV.py**
* *Outputs a CSV file to submit on Kaggle evaluating the model on the test set*
* *Only works for the two ensembles mentioned in the write-up (uncomment to test each classifier)*

**bin_naive_bayes.py**
*Implements binary naive bayes using Bing Liu's opinion lexicon (both positive and negative words) and nltk.corpus list of words from movie reviews as features
*Outputs accuracy, error rate, precision, recall, and specificity.

**bin_naive_bayes_bl.py**
*Implements binary naive bayes using Bing Liu's opinion lexicon (both positive and negative words) as features
*Outputs accuracy, error rate, precision, recall, and specificity.

**bin_naive_bayes_pos.py**
*Implements binary naive bayes using Bing Liu's opinion lexicon (only positive words) as features
*Outputs accuracy, error rate, precision, recall, and specificity.

**bin_naive_bayes_neg.py**
*Implements binary naive bayes using Bing Liu's opinion lexicon (only negative words) as features
*Outputs accuracy, error rate, precision, recall, and specificity.

**make_plot.py**
*Plots accuracy of binary naive bayes for different feature sets.


