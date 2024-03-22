import pandas as pd
import numpy as np
import get_data_utils
from dotenv import load_dotenv
import os
from openai import OpenAI

survivors_links, season_names = get_data_utils.scrape_all_seasons()

seasons_df, players_names, players_links = get_data_utils.scrape_each_season(survivors_links, season_names)

players_df = get_data_utils.get_players(players_names, players_links, season_names)

seasons_df.to_csv("../data/seasons.csv", index=False)
players_df.to_csv("../data/players.csv", index = False)

# seasons_df = pd.read_csv("seasons.csv")
# players_df = pd.read_csv("players.csv")

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

occupation_embeddings = get_data_utils.get_occupation_embeddings(all_df["occupation"], client)
all_df["occupation_embedding"] = occupation_embeddings

description_embeddings = get_data_utils.get_description_embeddings(all_df["player_description"], client)
all_df["description_embedding"] = description_embeddings

all_df.to_csv("../data/final.csv", index=False)
