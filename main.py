import pandas as pd
import time
from langdetect import detect
from nltk.corpus import stopwords
import itertools


def get_dataset(path):
    dataset = pd.read_csv(path, delimiter=';')
    dataset.drop(['tweetDate', 'tweetId'], axis=1, inplace=True)
    return dataset[dataset['tweetText'].notna()]


def generate_train_test(df, percentage=0.2):
    # Separem segons el sentiment del tweet
    df_t1 = df[df['sentimentLabel'] == 1]
    df_t0 = df[df['sentimentLabel'] == 0]

    # Mostrejem aleatoriament segons el percentatge train-test per cada subset
    t1_train = df_t1.sample(frac=percentage)
    t1_test = df_t1.drop(t1_train.index)

    t0_train = df_t0.sample(frac=percentage)
    t0_test = df_t0.drop(t0_train.index)

    # Retornem els conjunts train-test random
    train = pd.concat([t1_train, t0_train]).sample(frac=1)
    test = pd.concat([t1_test, t0_test]).sample(frac=1)

    return train, test


def generate_dict(train_df, remove_words=True):
    dict_words = {}
    a = False
    for r in train_df.to_dict('records'):
        for w in r['tweetText'].split():
            a = False
            try:
                dict_words[w][r['sentimentLabel']] += 1
            except:
                dict_words[w] = [0, 0]
                dict_words[w][r['sentimentLabel']] += 1

    if remove_words:
        stopW_en = stopwords.words('english')
        for k in stopW_en:
            dict_words.pop(k, None)

    return dict_words


def split_dict(d, mida_dict):
    d_f = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
    d_f = dict(itertools.islice(d_f.items(), int(mida_dict * len(d_f))))
    return d_f


def detect_lang(train_df):
    d_lang = []
    for index, row in train_df['tweetText'].iteritems():
        try:
            language = detect(row)
        except:
            language = "error"
            print("This row throws and error:", row)
        d_lang.append(language)
    return d_lang


def calc_individual_probs(n_train_0, n_train_1, d):
    probcond_d = d
    for k, v in d.items():
        probcond_d[k][0] /= n_train_0
        probcond_d[k][1] /= n_train_1

    return probcond_d


def test_model(Psent0, Psent1, test, probcond_d):
    t = [0,0]   # [TN,TP]
    f = [0,0]   # [FN,FP]

    for r in test.to_dict('records'):
        P_0 = Psent0
        P_1 = Psent1
        for w in r['tweetText'].split():
            if w in probcond_d:
                P_0 *= probcond_d[w][0]
                P_1 *= probcond_d[w][1]
        probs = [P_0, P_1]
        if probs.index(max(probs)) == r['sentimentLabel']:
            t[r['sentimentLabel']] += 1
        else:
            f[r['sentimentLabel']] += 1
    tn, tp = t
    fn, fp = f
    return tp, tn, fp, fn


def test_model_laplace_add1(Psent0, Psent1, test, train, probcond_d):
    t = [0, 0]  # [TN,TP]
    f = [0, 0]  # [FN,FP]

    n_train_1 = len(train[train['sentimentLabel'] == 1])
    n_train_0 = len(train[train['sentimentLabel'] == 0])

    for r in test.to_dict('records'):
        P_0 = Psent0
        P_1 = Psent1
        for w in r['tweetText'].split():
            if w in probcond_d:
                P_0 *= probcond_d[w][0]
                P_1 *= probcond_d[w][1]
            else:
                # laplace add-1
                P_0 *= 1 / n_train_0
                P_1 *= 1 / n_train_1
        probs = [P_0, P_1]
        if probs.index(max(probs)) == r['sentimentLabel']:
            t[r['sentimentLabel']] += 1
        else:
            f[r['sentimentLabel']] += 1
    tn, tp = t
    fn, fp = f
    return tp, tn, fp, fn


def test_model_laplace_smoothing(Psent0, Psent1, test, train, d, probcond_d, alpha):
    t = [0, 0]  # [TN,TP]
    f = [0, 0]  # [FN,FP]

    n_train_1 = len(train[train['sentimentLabel'] == 1])
    n_train_0 = len(train[train['sentimentLabel'] == 0])

    for r in test.to_dict('records'):
        P_0 = Psent0
        P_1 = Psent1
        for w in r['tweetText'].split():
            # laplace smoothing
            if w in probcond_d:
                P_0 *= (d[w][0] + alpha) / (n_train_0 + (alpha * 2))
                P_1 *= (d[w][1] + alpha) / (n_train_1 + (alpha * 2))
            else:
                P_0 *= (0 + alpha) / (n_train_0 + (alpha * 2))
                P_1 *= (0 + alpha) / (n_train_1 + (alpha * 2))
        probs = [P_0, P_1]
        if probs.index(max(probs)) == r['sentimentLabel']:
            t[r['sentimentLabel']] += 1
        else:
            f[r['sentimentLabel']] += 1
    tn, tp = t
    fn, fp = f
    return tp, tn, fp, fn


def naivebayes(train, test, mida_dict=1.0, smoothing="None", alpha=1):
    # train the model
    d = generate_dict(train, True)
    if mida_dict < 1:
        d = split_dict(d, mida_dict)

    n_train_1 = len(train[train['sentimentLabel'] == 1])
    n_train_0 = len(train[train['sentimentLabel'] == 0])

    probcond = calc_individual_probs(n_train_0, n_train_1, d)

    Psent1 = len(train[train['sentimentLabel'] == 1]) / len(train)
    Psent0 = len(train[train['sentimentLabel'] == 0]) / len(train)

    # test the model
    if smoothing == "None":
        tp, tn, fp, fn = test_model(Psent0, Psent1, test, probcond)
    elif smoothing == "laplace_add1":
        # laplace add-1 adds a standard count of [1+, 1-]
        tp, tn, fp, fn = test_model_laplace_add1(Psent0, Psent1, test, train, probcond)
    elif smoothing == "laplace_smoothing":
        Psent0 = (len(train[train['sentimentLabel'] == 0]) + alpha) / (len(train) + alpha * 2)
        Psent1 = (len(train[train['sentimentLabel'] == 1]) + alpha) / (len(train) + alpha * 2)
        tp, tn, fp, fn = test_model_laplace_smoothing(Psent0, Psent1, test, train, d, probcond, alpha)

    return tp, tn, fp, fn


def k_fold_validation(k, train_, smoothing="None"):
    print("~~~~~~~ K-fold validation with k={} ~~~~~~~~".format(k))

    eval_scores = []
    n_rows = int(len(train_) / k)

    train = train_
    fold_list = []

    print("Validant amb el training set...")
    for i in range(1, k + 1):
        fold = train.sample(n_rows)
        train = train.drop(fold.index)
        fold_list.append(fold)

    for idx in range(0, k):
        t0 = time.time()
        fold_test = fold_list[idx]
        fold_train = pd.concat([x for ind, x in enumerate(fold_list) if ind != idx])

        tp, tn, fp, fn = naivebayes(fold_train, fold_test, mida_dict, smoothing, alpha=1)
        acc = (tp + tn) / (tp + tn + fp + fn)

        eval_scores.append(acc)

        print("k -> {}\taccuracy = {:0.2f}\tt={:0.2f}".format(idx+1, acc,time.time() - t0))

    print("MEAN SCORE : {:0.2f}".format(sum(eval_scores) / len(eval_scores)))


def getMetricsDictTrain(path, smoothing="None"):
    dict_vals = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
    train_vals = [0.1, 0.2, 0.4, 0.6, 0.8]
    df = get_dataset(path)
    res = {}
    for dict_percentage in dict_vals:
        res[dict_percentage] = {}
        for train_percentage in train_vals:
            t0 = time.time()
            train, test = generate_train_test(df, train_percentage)
            tp, tn, fp, fn = naivebayes(train, test, dict_percentage, smoothing, alpha=1)
            tf = time.time() - t0
            acc = (tp + tn) / (tp + tn + fp + fn)
            res[dict_percentage][train_percentage] = [acc, tf]
    return res


if __name__ == '__main__':
    path = 'data/FinalStemmedSentimentAnalysisDataset.csv'

    """
    CROSSVALIDATION ESTÁ COMENTAT
    """

    '''PARAMS'''
    k = 4
    percentatge_crossvalidation = 0.2       # percentatge del training set que es pasara a la funcio de crossval
    percentage_naive = 0.2
    mida_dict = 0.05                        # assignar 1 per a treballar sense modificar la mida del diccionari
    smoothing = "None"
    # smoothing = "laplace_add1"
    # smoothing = "laplace_smoothing"
    alpha = 1

    ###########################################################################################
    #   CROSSVALIDATION K FOLD
    #
    #   Amb aquesta part del codi es fà la validació del model amb el training set
    #   utilitzant la tècnica de k-fold. En la seguent secció farem el Naive Bayes
    #   amb el testing-train set reals com a ultima fase del crossvalidation.
    #
    ###########################################################################################

    '''FUNCIONS'''
    t_crossvalidation = time.time()
    df = get_dataset(path)
    train, test = generate_train_test(df, percentatge_crossvalidation)
    k_fold_validation(k, train, smoothing)
    print("~~~~ Temps Crossvalidation : {:0.2f}s ~~~~\n".format(time.time() - t_crossvalidation))


    ###########################################################################################
    #   NAIVE BAYES
    #
    #   Amb aquest bloc es farà tota la pipeline de la construcció del model de Naive Bayes.
    #   Hi ha diferents paràmetres a configurar per probar diferents configuracions, bé sigui
    #   el smoothing, la mida del diccionari o el training set percentage.
    #
    #   Hi ha 3 possibles configutació del mètode de Naive Bayes possibles:
    #
    #          - naivebayes(train, test, smoothing="None", alpha=1, mida_dict)
    #               Aquest mètode si no troba una paraula en el diccionari la ignorarà
    #
    #          - naivebayes(train, test, smoothing="laplace_add1", alpha=1, mida_dict)
    #               Aquest mètode aplica una tècnica simple de smoothing "laplace add 1" que
    #               dona per suposat que TOTES les paraules han estat vistes amb una freqüencia
    #               estàndard.
    #               En el nostre cas l'haurem vist 2 vegades, 1 per cada possible outcome de la
    #               variable target sentimetnLabel.
    #
    #          - naivebayes(train, test, smoothing="laplace_smoothing", alpha=1, mida_dict)
    #               Aquest mètode aplica una tècnica de Laplace Smoothing estudiada a classe amb
    #               el paràmetre alpha.
    #
    #############################################################################################

    print("~~~~~ Naive Bayes Classification [percentage = {}, smoothing = {}, mida_dict = {}] ~~~~~"
          .format(percentage_naive, smoothing, mida_dict))
    t_naive = time.time()

    # Llegir i fer split
    df = get_dataset(path)
    train, test = generate_train_test(df, percentage_naive)

    # train model and test
    tp, tn, fp, fn = naivebayes(train, test, mida_dict, smoothing, alpha=1)

    # Calcul de mètriques
    acc = (tp + tn) / (tp + tn + fp + fn)
    # prec = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1score = 2 * (recall * prec) / (recall + prec)
    print("TP: {}   FP: {}   FN: {}   TN: {}".format(tp, fp, fn, tn))
    print("Accuracy: {:0.2f}".format(acc))

    print("~~~~ Temps Naive Bayes : {:0.2f}s ~~~~".format(time.time() - t_naive))
