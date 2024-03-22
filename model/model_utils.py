import numpy as np
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def preprocess_data(
    data: pd.DataFrame, 
    test_seasons: list,
    non_x_cols: list = ["player_link", "player_name", "season_name", "birthdate", "hometown", "occupation",
                "player_description", "season_link", "winner", "days", "start_date", "end_date", "num_castaways",
                "is_winner"], 
    y_col: str = "is_winner"):
    
    data_copy = data.copy()

    data_copy["occupation_embedding"] = data_copy["occupation_embedding"].apply(ast.literal_eval)
    data_copy["description_embedding"] = data_copy["description_embedding"].apply(ast.literal_eval)

    occ = pd.DataFrame(data_copy["occupation_embedding"].to_list(), columns=[f"occ_emb{i}" for i in range(len(data_copy["occupation_embedding"][0]))])
    desc = pd.DataFrame(data_copy["description_embedding"].to_list(), columns=[f"desc_emb{i}" for i in range(len(data_copy["description_embedding"][0]))])

    data_prep = pd.concat([data_copy.drop(columns=["occupation_embedding", "description_embedding"]), occ, desc], axis=1)


    X = data_prep.drop(columns=non_x_cols)
    y = data_prep[y_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # X_test = X[X["season_no"] == 46]
    # X_train = X[X["season_no"] != 46]

    # Change this to test_seasons and also, with NAs in the column, it will make the numbers strings - fix this
    X_test_scaled = X_scaled[X["season_no"].isin(test_seasons)]
    X_train_scaled = X_scaled[~X["season_no"].isin(test_seasons)]

    X_test_scaled = X_test_scaled.to_numpy()
    X_train_scaled = X_train_scaled.to_numpy()


    y_train = y[~X["season_no"].isin(test_seasons)]
    
    return X_train_scaled, X_test_scaled, y_train

######### SVM

def model_predict(data: pd.DataFrame, test_seasons: list, bootstrap: int = 1, sort: bool=True):
    test = data[data["season_no"].isin(test_seasons)]

    X_train_scaled, X_test_scaled, y_train = preprocess_data(data, test_seasons)

    bootstrap_probs = []
    for i in range(bootstrap):
        
        linear_svm = SVC(C=10, kernel="linear", probability=True, class_weight={"Winner":15, "Non-Winner":1})
        linear_svm.fit(X_train_scaled, y_train)
        linear_svm_preds = linear_svm.predict_proba(X_test_scaled)
        
        winner_prob = [i[1] for i in linear_svm_preds]
        winner_prob = pd.Series(winner_prob, name="winner_prob")
        winner_prob.index = test.index
        
        new_winner_prob = winner_prob/winner_prob.groupby(test["season_no"]).transform("sum").tolist()
        bootstrap_probs.append(new_winner_prob)
    
    # Convert the list of lists into a NumPy array for efficient computation
    bootstrapping_array = np.array(bootstrap_probs)
    
    test["winning_probability"] = np.mean(bootstrapping_array, axis=0)
    test["std"] = np.std(bootstrapping_array, axis=0)

    test = test[["player_name", "winning_probability", "std"]]
    if sort:
        test = test.sort_values("winning_probability", ascending=False)
    return test


########## Prediction plot

# final_predictions.to_csv("season_46_preds.csv")

# plt.bar(final_predictions["player_name"], final_predictions["winning_probability"])
# plt.xticks(rotation=90)
# plt.show()

# Plotting
def plot_preds(player_names, predictions, season, std=None):
    is_std = False if std is None else True
    plt.figure(figsize=(10, 6))
    if is_std:
        bars = plt.bar(player_names, predictions * 100, yerr=std*100, error_kw={"capsize":3})  # Multiplying by 100 to get percentage
    else:
        bars = plt.bar(player_names, predictions * 100)  # Multiplying by 100 to get percentage

    # Adding percentages above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom')  # Adjust rotation as needed

    # Formatting
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Player Name')
    plt.ylabel('Winning Probability (%)')
    plt.title(f'Predicted Winning Probability for Survivor {season}')
    plt.tight_layout()

    return plt
    # Show the plot
    # plt.show()