import pandas as pd
import numpy as np
import model_utils

final_data = pd.read_csv("../data/final.csv")

final_data = final_data[final_data["season_no"] != 46]

final_45 = model_utils.model_predict(final_data, [45], bootstrap=1000, sort=True)

final_45.to_csv("../data/season_45_boot_preds.csv", index=False)

final_45 = pd.read_csv("../data/season_45_boot_preds.csv")

plot = model_utils.plot_preds(final_45["player_name"], final_45["winning_probability"], 45, final_45["std"])

plot.savefig("../images/plots/final_45.png")