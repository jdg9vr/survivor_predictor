import torch
import numpy as np
from hyperfast import HyperFastClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import ast
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import keras
from keras import ops
from keras import layers
import tensorflow as tf

# Read data
data = pd.read_csv("final.csv")

data["occupation_embedding"] = data["occupation_embedding"].apply(ast.literal_eval)
data["description_embedding"] = data["description_embedding"].apply(ast.literal_eval)

occ = pd.DataFrame(data["occupation_embedding"].to_list(), columns=[f"occ_emb{i}" for i in range(len(data["occupation_embedding"][0]))])
desc = pd.DataFrame(data["description_embedding"].to_list(), columns=[f"desc_emb{i}" for i in range(len(data["description_embedding"][0]))])

data_prep = pd.concat([data.drop(columns=["occupation_embedding", "description_embedding"]), occ, desc], axis=1)



# Train-test split
non_x_cols = ["player_link", "player_name", "season_name", "birthdate", "hometown", "occupation",
              "player_description", "season_link", "winner", "days", "start_date", "end_date", "num_castaways",
              "is_winner"]
y_col = "is_winner"


X = data_prep.drop(columns=non_x_cols)
y = data_prep[y_col]



random.seed(12)
test_seasons = random.sample(range(1, 46), 6)

X_test = X[X["season_no"].isin(test_seasons)]
X_train = X[~X["season_no"].isin(test_seasons)]

y_test = y[X["season_no"].isin(test_seasons)]
y_train = y[~X["season_no"].isin(test_seasons)]


def get_test_ranks(X_test, y_test, prediction_prob):
    winner_prob = pd.Series(prediction_prob, name="winner_prob")

    winner_prob.index = y_test.index

    full_test = pd.concat([X_test, y_test, winner_prob], axis=1)

    winner_ranks = []
    for i in full_test["season_no"].unique():
        season_test = full_test[full_test["season_no"] == i]
        season_test = season_test.sort_values("winner_prob", ascending=False).reset_index(drop=True)
        real_winner_predicted_rank = season_test[season_test["is_winner"] == "Winner"].index[0]+1
        winner_ranks.append(real_winner_predicted_rank)

    return winner_ranks

##### HYPERFAST NEURAL NETWORK

# Set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize HyperFast
model = HyperFastClassifier(device=device)

# Generate a target network and make predictions
model.fit(X_train, y_train)
predictions_proba = model.predict_proba(X_test)
predictions = model.predict(X_test)

predictions_single = [i[1] for i in predictions_proba]

hyperfast_ranks = get_test_ranks(X_test, y_test, predictions_single)

###### LOGISTIC REGRESSION

log_reg = LogisticRegression(random_state=0).fit(X_train, y_train)
winner_prob = [i[1] for i in log_reg.predict_proba(X_test)]

log_reg_ranks = get_test_ranks(X_test, y_test, winner_prob)


###### KNN

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=10, weights="distance")
knn.fit(X_train_scaled, y_train)

knn_k10_predicted_probs = knn.predict_proba(X_test_scaled)

k_values = [i for i in range (1,200)]
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    score = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    scores.append(np.mean(score))



knn = KNeighborsClassifier(n_neighbors=200, weights="distance")
knn.fit(X_train_scaled, y_train)

knn_preds = knn.predict_proba(X_test_scaled)
winner_prob = [i[1] for i in knn_preds]

knn_ranks = get_test_ranks(X_test, y_test, winner_prob)


######### SVM

C_vals = [1, 10, 100, 1000, 10000]
scores = []
for C in C_vals:
    this_model = SVC(kernel="linear", probability=True, class_weight={"Winner":15, "Non-Winner":1})
    score = cross_val_score(this_model, X_train_scaled, y_train, cv=5)
    scores.append(np.mean(score))

linear_svm = SVC(C=10, kernel="linear", probability=True, class_weight={"Winner":15, "Non-Winner":1})
linear_svm.fit(X_train_scaled, y_train)
linear_svm_preds = linear_svm.predict_proba(X_test_scaled)

winner_prob = [i[1] for i in linear_svm_preds]

linear_svm_ranks = get_test_ranks(X_test, y_test, winner_prob)


poly3_svm = SVC(C=10, kernel="poly", degree=3, probability=True, class_weight={"Winner":15, "Non-Winner":1})
poly3_svm.fit(X_train_scaled, y_train)
poly3_svm_preds = poly3_svm.predict_proba(X_test_scaled)

winner_prob = [i[1] for i in poly3_svm_preds]

poly3_svm_ranks = get_test_ranks(X_test, y_test, winner_prob)





poly3_svm = SVC(C=100, kernel="poly", degree=5, probability=True, class_weight={"Winner":15, "Non-Winner":1}, random_state=134)
poly3_svm.fit(X_train_scaled, y_train)
poly3_svm_preds = poly3_svm.predict_proba(X_test_scaled)

winner_prob = [i[1] for i in poly3_svm_preds]

poly3_svm_ranks = get_test_ranks(X_test, y_test, winner_prob)


######### TRANSFORMER


train_seasons = [i for i in range(1, 46)]
for i in test_seasons:
    train_seasons.remove(i)

random.seed(2725)
train_mini_seasons = random.sample(train_seasons, round(len(train_seasons)/2))

X_train_mini = X_train[X_train["season_no"].isin(train_mini_seasons)]
X_val = X_train[~X_train["season_no"].isin(train_mini_seasons)]

scaler = StandardScaler()
X_train_mini_scaled = scaler.fit_transform(X_train_mini)
X_val_scaled = scaler.fit_transform(X_val)

y_train_mini = y_train[X_train["season_no"].isin(train_mini_seasons)]
y_val = y_train[~X_train["season_no"].isin(train_mini_seasons)]

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, rate=0.1):
        super().__init__()
        self.ffn1 = keras.Sequential(
            [layers.Dense(embed_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        # self.ffn2 = keras.Sequential(
        #     [layers.Dense(embed_dim, activation="relu"), layers.Dense(ff_dim1),]
        # )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.ffn1(inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn1(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

ff_dim1 = 212  # Hidden layer size in feed forward network inside transformer
ff_dim2 = 128

inputs = layers.Input(shape=(X_train_mini_scaled.shape[1],))
transformer_block = TransformerBlock(X_train_mini_scaled.shape[1])
x = transformer_block(inputs)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

y_train_mini = y_train_mini.replace({"Winner":1, "Non-Winner":0}).to_numpy()
y_val = y_val.replace({"Winner":1, "Non-Winner":0}).to_numpy()

y_train_mini = keras.utils.to_categorical(y_train_mini)
y_val = keras.utils.to_categorical(y_val)

X_train_mini = X_train_mini.to_numpy()
X_val = X_val.to_numpy()

# (asdfg, asdfg2), (asdfg3, asdfg4) = keras.datasets.imdb.load_data(num_words=100)

tf.config.run_functions_eagerly(True)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], run_eagerly=True)
history = model.fit(
    X_train_mini, y_train_mini, batch_size=32, epochs=100, validation_data=(X_val, y_val)
)

