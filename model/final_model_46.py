import pandas as pd
import numpy as np
import model_utils

final_data = pd.read_csv("../data/final.csv")

final_46 = model_utils.model_predict(final_data, [46], bootstrap=1000, sort=True)

final_46.to_csv("../data/season_46_boot_preds.csv", index=False)

# final_46 = pd.read_csv("../data/season_46_boot_preds.csv")

plot = model_utils.plot_preds(final_46["player_name"], final_46["winning_probability"], 46, final_46["std"])

plot.savefig("../images/plots/final_46.png")