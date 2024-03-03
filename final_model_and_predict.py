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
import matplotlib.pyplot as plt

# Merge Season 46 with all data

data = pd.read_csv("final.csv")
season_46 = pd.read_csv("season_46.csv")

data = pd.concat([data, season_46]).reset_index(drop=True)


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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_test = X[X["season_no"] == 46]
X_train = X[X["season_no"] != 46]

X_test_scaled = X_scaled[X["season_no"] == 46]
X_train_scaled = X_scaled[X["season_no"] != 46]

X_test_scaled = X_test_scaled.to_numpy()
X_train_scaled = X_train_scaled.to_numpy()


y_train = y[X["season_no"] != 46]



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

linear_svm_preds


########## Prediction dataframe

winner_prob = [i[1] for i in linear_svm_preds]
winner_prob = pd.Series(winner_prob, name="winner_prob")

winner_prob = winner_prob/winner_prob.sum()

winner_prob.index = X_test.index

data_prep["winning_probability"] = winner_prob

final_predictions = data_prep.loc[X["season_no"] == 46, ["player_name", "winning_probability"]]

final_predictions = final_predictions.sort_values("winning_probability", ascending=False)

final_predictions.to_csv("season_46_preds.csv")

plt.bar(final_predictions["player_name"], final_predictions["winning_probability"])
plt.xticks(rotation=90)
plt.show()


# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(final_predictions['player_name'], final_predictions['winning_probability'] * 100)  # Multiplying by 100 to get percentage

# Adding percentages above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
             f'{height:.2f}%', ha='center', va='bottom', rotation=45)  # Adjust rotation as needed

# Formatting
plt.xticks(rotation=45, ha='right')
plt.xlabel('Player Name')
plt.ylabel('Winning Probability (%)')
plt.title('Winning Probability for Each Player')
plt.tight_layout()

# Show the plot
plt.show()