# Twitter Sentiment Analysis
NLP Sentiment Analysis on tweets using a Bayesian Network implemented from scratch without any library.

## Introduction
### Naive Bayes assumption of Independence
Naive Bayes emerges as an alternative to Joint Distribution, it is based on Bayes' assumption that all attributes are conditionally independent. This assumption, more formally called the Bayes Theorem, may seem like a crazy assumption, but it has become very powerful over time because it allows us to use computational power. Over the years it has become a standalone as a robust classifier that with little training and large databases can give many good results.

### Objective
For the project we are given a dataset which is more than 1M tweets.
These tweets have already been processed with a "Lancaster" stemming algorithm that extracts the lexeme from the words to reduce the size of the dataset and concentrate the entire workload on the morphemes.
For example, if we had the following words `<programmer, program, programmatically, programmable>` the Lancaster algorithm would reduce all words to one: `<program>`.

The dataset follows the following structure:
![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/query_structure.png)

Our algorithm must classify tweets according to the sentimentLabel attribute, that is, it must predict whether a tweet is positive (sentimentLabel = 1) or negative (sentimentLabel = 0).
We need to look at how they affect the size of the training dictionary and the train-test split.
Finally, we will also need to examine and implement some smoothing method to avoid the probabilities at 0.

## Proposal
The proposal is incremental, meaning that we will first start with a basic bayesian network and then we will be adding more features and capabilities to our model.

### Basic Bayesian Network
To implement a first Bayesian network, I followed the steps below:
-  **Feature engineering**: In this step we will read the data, do a quick analysis to detect anomalies and divide it into train and test sets.
-  **Train the model**: In this section we will generate a dictionary with the words that we find in the training set. We will get rid of meaningless words with the help of stopwords.
We will also calculate the probabilities of each word.
-  **Predictions**: We will test our training with the new data set, test. This is where we need to apply smoothing if we don't want to ignore words that aren't part of the dictionary.
-  **Metrics**: Once we have done the classification we will be able to extract different metrics to determine the performance of our model.

#### Feature Engineering: Train-test split
##### Read dataset
In this step we read the dataset and get rid of the columns <tweetId, tweetDate> since we will not use them for anything.
We will also eliminate the rows where we find some NaN to avoid possible a posteriori complications, 22 rows in particular are insignificant considering that we have more than 1M of "memories".
##### Train-split test
We will specify the percentage of the training set.
To do this in a balanced way, we will first separate the dataset according to the target sentimentLabel variable, thus obtaining two subsets, the positive and the negative.

```python
df[df['sentimentLabel'] == 1]    # All the tweets with 'sentimentLabel' equal to one
```

Then sample randomly according to the specified train-test percentage. Form train and test sets for separate positive and negative tweets.

```python
t1_train = df_t1.sample(frac=percentage)
t1_test = df_t1.drop(t1_train.index)
```

Finally, we will combine the two positive and negative train-test subsets. Using the sample function with 100% percentage to shuffle all rows.

```python
train = pd.concat([t1_train, t0_train]).sample(frac=1)
test = pd.concat([t1_test, t0_test]).sample(frac=1)
```

#### Training: Dictionaries generation and "stopwords"
##### Dictionaries generation
Model training will be done by calculating the probabilities of each word.
To do this we must first calculate the frequency of each word according to its target attribute. We will save it in a dictionary called dict_words.

```python
dict_words = {“<word>”: [negative_count, positive_count]}
```

We will use this dictionary of counts to create the dictionary where we will calculate the probabilities of each word in the dictionary. We will save it in a dictionary called prob_cond.

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/probs_formula.png)

##### Stopwords
The NLTK library's stopwords feature has been used to detect words that have no meaning such as pronouns or prepositions.
Stopwords is available in many languages, which is why an auxiliary function has been made to analyze the language of tweets (detect_lang).

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/langs.png)

As we can see, the vast majority of tweets (840545) are in English, so we will remove all meaningless words with the help of the list of words generated by the stopwords function.

#### Prediction
With dictionaries we can start making predictions.
For each tweet, we look up the words in our dictionary of probabilities and multiply them.
With the help of Bayes's assumption, we'll calculate the largest conditional probability accumulation for each word in the tweet so we can predict which outcome it belongs to, whether the tweet is positive or negative.

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/bayes_formula.png)

In this section, we will ignore words that do not appear in the training dictionary. We ignore them to avoid generating probabilities of 0.

#### Metrics
Given  `True Positive (TP)`, `True Negative (TN)`, `False Positive (FP)` and `False Negative (FN)` we can define some metrics as so:
```python
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1Score = 2 * (Recall * Precision) / (Recall + Precision)
```

Accuracy tells us the success rate of our model.
In binary classifiers, as is the case, the accuracy metric can be misleading when we have an unbalanced dataset in which there are more positive than negative tweets or vice versa.

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/balanced.png)

But in this case, if we calculate the percentage of each outcome in the database we see that the dataset is balanced so the accuracy will be a good metric.

### Crossvalidation
Model validation is a widely used technique for quickly detecting overfitting, when the model is too close to training data or underfitting, when the model fails to achieve the minimum and necessary learning in training.

It also serves to validate that the results are consistent and reliable for different train-test splits of the data.

(*) Keep in mind that cross-validation is done using only the entire training set, doing so with the testing set would be a mistake.

#### K-Fold technique
A good technique that allows us to do cross-validation without a large load of resources is K-fold.

This technique divides the training dataset into k parts called folds. Of these k folds, 1 will be test and the remaining k-1 will be training. K experiments will be done so each fold will be tested once and finally we will get the crossvalidation score as the average of the accuracy obtained

To implement this first a specific function has been done in which the train dataset is divided into K partitions and with a loop we will make sure that each partition is used as a test once.

### Playing with dictionary sizes
To reduce the size of the dictionary, a function is created in which the word counter dictionary is sorted from most to least frequent and then with a given percentage as a parameter we split, leaving the percentage of words. more frequent.

In this case mida_dict will be the value as a percentage of the most frequent words we want to keep.

With sorted we do the descending order of the doubles of appearance, both positive and negative, and with the help of itertools we can get rid of a part of the dictionary.

#### Accuracy and runtime
As you can see, we have a balanced set, so Accuracy is a good estimator of the model's performance.

To carry out the study of this section, a getMetricsDictTrain function has been carried out which, with a list of different dictionary sizes and training set, will perform tests with the combination of all sizes and we will save accuracy and execution time. of each possible combination of values.

```python
sizes_dict = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
sizes_train = [0.1, 0.2, 0.4, 0.6, 0.8]
```

It will return the results with the following structure, where the first level of keys will be the size of the dictionary and the second level of keys will be the size of the learning set:

```python
results_dict[size_dict][size_train] = accuracy
```

Would display something like this:

```python
d = {
  # Dict Sizes
  0.001:
  {
    # Train Sizes
    0.1: [0.63, 21.08],
    0.2: [0.63, 22.50],
    0.4: [0.63, 21.08],
    ...
  },
  0.01:
  {
    ...
  },
  ...
}
```

### Smoothing Techniques
So far we are ignoring words that do not appear in our dictionary. But there must be some method to quantify the rest of the words as well.
So what do we do with words that are not in our dictionary?
In the naive bayes function there is a "smoothing" parameter that depending on the value we give it will apply (or not) any of the following settings:
- `smoothing = “None”`
  
  Unknown words will be ignored to avoid generating probabilities of value 0.

- `smoothing = “laplace_add1”`

  Apply a simple "laplace add 1" smoothing technique that assumes all
words have been viewed with standard frequency.
  In our case we have seen it 2 times, 1 for each possible outcome of the target sentimentLabel variable.
  In our case the value of the unseen words in the generated dictionary will be:
  ```python
  dict_words = { "<word>": [1, 1] }
  ```
  ![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/prob_laplace_add1.png)

- `smoothing=“laplace_smoothing”`
  
  Apply the Laplace Smoothing technique with the alpha parameter, as explained in
class.
  We will calculate the probabilities with the following formula:
  
  ![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/prob_laplace_smooth.png)
  
  Alpha multiplies the possible outcomes of the target variable (2).
  
## Results
### Playing with dictionary sizes
If we run the algorithm without modifying the dictionary, with a standard train percentage of 0.6 and with the same learning set we get the following results:

```
NAIVE BAYES CLASSIFICATION 
(train set = 60%, smoothing = None, dictionary = 100%)

TP: 212278
FP: 100376
FN: 65861
TN: 247197

Accuracy: 0.73

Time: 19.60s
```

Now it's time to look at how the algorithm will work for different dictionary and training set sizes.

We will focus on the results dictionary with the different metrics, the accuracy and runtime for different value combinations that the getMetricsDictTrain function will return.

Let's try the values:
```python
sizes_dict = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
sizes_train = [0.1, 0.2, 0.4, 0.6, 0.8]
```

We get the following results:

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/res1.png)

We can see that as long as the dictionary is at least 1% it will give good results. From there it will decrease the accuracy so a good dictionary size range would be between 1% - 10%.

We note that with 5% of the most common knowledge, we will get the best results.

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/res2.png)

On the other hand, with train set we can't see much difference in terms of accuracy. There is a 0.01 accuracy between the minimum and maximum.

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/res3.png)

In order to finish concretizing these deductions we would have to observe the results with the execution time:

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/res4.png)

With this graph we can see how increasing the size of the dictionary and the training set is proportional to a longer run time. But the increase in the train set seems to have more weight over time than the increase in the dictionary.

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/res5.png)

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/res6.png)

**Therefore the most optimal parameters will be a dictionary size of 5% and a training set of 20%.**

### Crossvalidation
With crossvalidation we can specify the dimension of the training set with which we will do the corresponding validation.

We observe that if we cross-validate with a training dataset of 20% (312,856 rows), with a K of value 10, we obtain the same accuracy as if we did it with a training dataset of 80% (1,251,424 rows).

With the only difference that with 20% of train the execution time is 36.71s and with 80% of train the execution time is 135.28s.

```
K-fold validation with 312,856 rows (20% training set)
k = 10

k = 1   ->  accuracy = 0.74, t = 3.34
k = 2   ->  accuracy = 0.74, t = 3.32
k = 3   ->  accuracy = 0.74, t = 3.22
k = 4   ->  accuracy = 0.74, t = 3.26
k = 5   ->  accuracy = 0.74, t = 3.25
k = 6   ->  accuracy = 0.75, t = 3.18
k = 7   ->  accuracy = 8.74, t = 3.18
k = 8   ->  accuracy = 0.74, t = 3.23
k = 9   ->  accuracy = 0.75, t = 3.16
k = 10  ->  accuracy = 0.75, t = 3.14

MEAN SCORE : 0.74

Total time Crossvalidation 36,71s
```

```
K-fold validation with 1,251,424 rows (80% training set)
k = 10

k = 1   ->  accuracy = 0.75, t = 13.08
k = 2   ->  accuracy = 0.75, t = 12.69
k = 3   ->  accuracy = 0.75, t = 12.77
k = 4   ->  accuracy = 0.75, t = 13.09
k = 5   ->  accuracy = 0.75, t = 12.67
k = 6   ->  accuracy = 0.75, t = 12.99
k = 7   ->  accuracy = 0.75, t = 13.09
k = 8   ->  accuracy = 0.75, t = 12.82
k = 9   ->  accuracy = 0.75, t = 12.85
k = 10  ->  accuracy = 0.75, t = 12.69

MEAN SCORE : 0.75

Total time Crossvalidation 135,28s
```

If we generalize this experiment we can observe in detail the variation of the accuracy for different values of the training set.

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/res7.png)

Similarly, we can look for the most optimal k to get good results.

If we keep the same percentage of 20% of training set, we observe that with a k of value 2 already gives results, therefore, with a k of value 5 it would already be enough to us.

```
K-fold validation with (20% training set)
k = 2

k = 1   ->  accuracy = 0.74, t = 2.68
k = 2   ->  accuracy = 0.74, t = 2.80

MEAN SCORE : 0.74

Total time Crossvalidation 9.67s
```

```
K-fold validation with (20% training set)
k = 4

k = 1   ->  accuracy = 0.74, t = 3.10
k = 2   ->  accuracy = 0.74, t = 3.05
k = 3   ->  accuracy = 0.74, t = 3.04
k = 4   ->  accuracy = 0.74, t = 3.15

MEAN SCORE : 0.74

Total time Crossvalidation 16.60s
```

So we have achieved optimal parameters for crossvalidation. Originally with 80% train and k = 10 the run time was 135.28s and now with the new train parameters of 20% and k = 4, to ensure reliability, the run time is 18.38s.

By analyzing the results we obtain a SpeedUp of 7.36 (736%) in the model validation.

Because with this experiment the original training set is used as the new total dataset of the problem, basically, when we reduce the training dataset from 80% to 20%, we are reducing the whole learning set of the problem.

Then this experiment shows us that we can also reduce the whole learning set, both train and test, when we apply Naive Bayes and still have good results but in less time.

This algorithm will work on both small and large scale.

### Smoothing Techniques
Until now, we were ignoring unknown words in the dictionary.
But with smoothing we can count these unknown words in our algorithm.

We run the program with the optimal parameters found in the previous section, 20% of the train set and 5% of the dictionary size.

- `smoothing = “None”`
  
  ```
  NAIVE BAYES CLASSIFICATION 
  (train set = 20%, smoothing = None, dictionary = 5%)

  TP: 407601
  FP: 217708
  FN: 100506
  TN: 525609

  Accuracy: 0.75

  Time: 17.68s
  ```

- `smoothing = “laplace_add1”`

  ```
  NAIVE BAYES CLASSIFICATION 
  (train set = 20%, smoothing = laplace_add1, dictionary = 5%)

  TP: 412892
  FP: 212417
  FN: 103684
  TN: 522431

  Accuracy: 0.75

  Time: 17.24s
  ```
  
  Laplace add 1 applies a simple smoothing technique in which it assumes that we have already seen any unknown word at a frequency of [1,1], that is, once positive and once negative.

  This, when doing the calculations with the probabilities, as we have so many words, gives us very small probabilities and that is why it hardly has weight in the results. Accuracy remains the same.

- `smoothing=“laplace_smoothing”`

  ```
  NAIVE BAYES CLASSIFICATION 
  (train set = 20%, smoothing = Laplace_smoothing, dictionary = 5%)

  TP: 490308
  FP: 135001
  FN: 272689
  TN: 353426

  Accuracy: 0.67

  Time: 22.78s
  ```
  
  This time, unlike the simple smoothing applied above, the odds of unfamiliar words carry more weight and are noticeable in the accuracy, which decreases.
  We analyze how the sizes of the dictionary and the train set affect the accuracy and execution time with this smoothing technique.

  With this new configuration with Laplace Smoothing instead of ignoring, we soften to take into account those cases where there are unknown words, we manage to stabilize the accuracy, as it varies virtually nothing from a minimum dictionary size of the '1% and with any size of the training set.

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/res8.png)

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/res9.png)

If we look at the execution times, it's much more tedious than ignoring it. It is a more realistic algorithm with the results as unknown words are taken into account, but generally slower.

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/res10.png)

![alt text](https://github.com/ggcr/Twitter-Sentiment-Analysis/blob/main/imgs/res11.png)

For laplace smoothing we will keep a low dictionary size and a low training set so we can run it in less time.

Generally, we do not get any real benefit from applying Laplace Smoothing, neither with the accuracy of the model nor with the execution time.

## Conclusions

### Problems
The algorithm can still be optimized quite a bit in terms of time. Small changes have been made throughout the practice where I saw that the program was running smoothly. However, a closer look can greatly optimize it.

It should also be noted that it is sometimes difficult to strike a balance between the readability of python code and making code optimal and fast.

Although it didn't give me any problems, tests could be inserted to make the code secure so I could test other datasets and experiment with the program.

### Final Remarks
From the experiments with the modification of the size of the dictionary and of the training set we can conclude, the same that has been commented in the introduction of the project; Naive Bayes gives robust enough results with little training.

With a training set of 20% and a dictionary size of 5% it will give the best results with an accuracy of 75%.

With the cross-validation experiment it can be concluded that it is a very fast algorithm to validate with k-fold, there has been no indication of overfitting or underfitting no matter how much we test with different parameters.

With a training set of 20% and a k of value 4 we will have greatly reduced the execution time and validating the model does not involve any effort.

In addition, this validation experiment has given us good indications that this algorithm works well with both small and large datasets, because since k-fold uses the training set to generate K partitions and 1 of them test, when we have reduced the training set, we were really reducing the whole learning set.

Smoothing techniques have not been very successful. In this issue, the model concentrates the weight of a tweet's sentiment on some specific keywords (those that have the most probabilistic weight in the dictionary) that are revealed during the training phase. Therefore, unseen words will be difficult to handle.

This model will give even better results in a real application if every certain period of time (days) we make you update your dictionary, learning new words as well as discovering new keywords and vice versa.
