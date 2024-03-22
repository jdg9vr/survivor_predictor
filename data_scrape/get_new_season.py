import pandas as pd
import numpy as np
import get_data_utils
from dotenv import load_dotenv
import os
from openai import OpenAI

season_name = "Survivor: 46"
link = "/wiki/Survivor_46"

seasons_df, players_names, players_links = get_data_utils.scrape_each_season([link], [season_name])

players_df = get_data_utils.get_players(players_names, players_links, [season_name])

seasons_df.to_csv("season_46.csv", index=False)
players_df.to_csv("players_46.csv", index = False)


all_df = get_data_utils.merge_and_clean(players_df, seasons_df)

load_dotenv()

# Get embeddings
openai_key = os.getenv('OPENAI_KEY')
client = OpenAI(api_key = openai_key)

# # Test
# response = client.embeddings.create(
#     input=all_df["occupation"][0],
#     model="text-embedding-3-large"
# )

occupation_embeddings = get_data_utils.get_occupation_embeddings(all_df["occupation"])
all_df["occupation_embedding"] = occupation_embeddings

description_embeddings = get_data_utils.get_description_embeddings(all_df["player_description"])
all_df["description_embeddings"] = description_embeddings

all_df.to_csv("final_46.csv", index=False)