import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class Bagging():
    def __init__(self, data, test, m):
        self.data = data
        self.test = test

        self._preprocess(self.data)
        self._preprocess(self.test)

        self._m = m
        self.models = []
        self.samples = []


    @staticmethod
    def _preprocess(df):
        """
        Transforms nominal data to ordinal and boolean
        :param df: df to transform
        :return:
        """
        df['result'] = np.where(df['survived'] == 'yes', 1, 0)
        df['is_female'] = np.where(df['gender'] == 'female', 1, 0)
        df['is_child'] = np.where(df['age'] == 'child', 1, 0)
        df['pclass'] = pd.Categorical(df['pclass'],
                                      ordered=True,
                                      categories=['1st', '2nd', '3rd', 'crew']
                                      ).codes

        df.drop(columns=['gender', 'age', 'survived'], inplace=True)

    def fit(self):
        """
        Trains the model based on dataset
        :return:
        """
        self.samples = self._get_samples()
        for i in range(self._m):
            X = self.samples[i].drop(['result'], axis=1)
            Y = self.samples[i]['result'].where(self.samples[i]['result'] == 1, 0)

            model = self._train_model(X, Y)
            self.models.append(model)

    def _get_samples(self):
        # Given a training set of size n, create m samples of size n
        # by drawing n examples from the original data, with replacement
        samples = []
        unique_rows = self.data.drop_duplicates()
        for i in range(self._m):
            #Each bootstrap sample will on average contain 63.2% of the
            # unique training examples, the rest are replicates
            sample = unique_rows.sample(frac=0.632)
            complementary_size = len(self.data) - len(sample)
            complementary = sample.sample(n=complementary_size, replace=True)
            samples.append(pd.concat([sample, complementary]))
        return samples

    def _train_model(self, X, Y):
        """
        Trains model based using Tree Stump based on X and Y
        :param X: Dataset without result parameter
        :param Y: Result parameter
        :return: Trained model
        """
        tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=1)
        model = tree_model.fit(X, Y)
        return model

    def predict(self):
        """
        Predict based on previous created models
        :return:
        """
        predictions = []
        for i, model in enumerate(self.models):
            prediction = model.predict(self.test.drop(['result'], axis=1))
            predictions.append(prediction)

        pred_matrix = np.array(predictions)

        #Combine the m resulting models using simple majority vote
        result = np.apply_along_axis(self._mode, 0, pred_matrix)
        self._output_result(result)

    @staticmethod
    def _mode(a):
        u, c = np.unique(a, return_counts=True)
        return u[c.argmax()]


    def _output_result(self, result):
        """
        Outputs the result
        :param result: prediction result
        :return:
        """
        self._output_success(result)

        self._clean_for_output()

        self.test.to_csv('titanikPrediction.csv', index=False)
        print(self.test)

    def _output_success(self, result):
        """
        Calculates the percentage of success based on prediction results
        :param result: prediction result
        :return:
        """
        self.test['pred'] = result
        self.test['corrects_rows'] = result == self.test.result
        corrects_rows = np.sum(self.test['corrects_rows'].astype(int))
        n = len(self.test.pred)
        print("Success: {}%".format(corrects_rows * 100 / n))

    def _clean_for_output(self):
        """
        Transform tests data as it was originally before pre-process
        :return:
        """
        self.test['survived'] = np.where(self.test['result'] == 1, 'yes', 'no')
        self.test['gender'] = np.where(self.test['is_female'] == 1, 'female', 'male')
        self.test['age'] = np.where(self.test['is_child'] == True, 'child', 'adult')
        self.test['pclass'] = self.test['pclass'].replace({0: '1st', 1: '2nd', 2: '3rd', 3: 'crew'})
        self.test['pred'] = np.where(self.test['pred'] == 1, 'yes', 'no')

        self.test.drop(columns=['result', 'is_female', 'is_child', 'result', 'corrects_rows'], inplace=True)


if __name__ == "__main__":
    b = Bagging(pd.read_csv('titanikData.csv'),
                pd.read_csv('titanikTest.csv', names=["pclass", "age", "gender", "survived"]), m=100)
    b.fit()
    b.predict()
