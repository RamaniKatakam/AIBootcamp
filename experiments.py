{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4b801e8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import load_iris\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X = load_iris().data\n",
    "y = load_iris().target\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)\n",
    "\n",
    "clf = DecisionTreeClassifier()\n",
    "fitted = clf.fit(X_train, y_train)\n",
    "\n",
    "y_pred = fitted.predict(X_test)\n",
    "\n",
    "score = accuracy_score(y_test, y_pred)\n",
    "print('Accuracy: ', score)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
