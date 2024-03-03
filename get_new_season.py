import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from openai import OpenAI


season_no_list = []
days_list = []
dates_list = []
start_date_list = []
end_date_list = []
castaways_num_list = []
players_names = []
players_links = []


season_name = "Survivor: 46"
link = "/wiki/Survivor_46"

# Get all the names and links for each player that season
single_season = requests.get("https://survivor.fandom.com" + link)

single_soup = BeautifulSoup(single_season.text)

season_no = single_soup.find("div", {"data-source":"season"}).text.replace("\n", "").replace("Season No.", "")
season_no_list.append(season_no)

days = single_soup.find("div", {"data-source":"days"}).text.replace("\n", "").replace("No. of Days", "")
days_list.append(days)

dates = single_soup.find("div", {"data-source":"filmingdates"}).text.replace("\n", "").replace("Filming Dates", "")
dates_list.append(dates)

castaways_num = single_soup.find("div", {"data-source":"survivors"}).text.replace("\n", "").replace("No. of Castaways", "")
castaways_num_list.append(castaways_num)

# Can check if a player has a birthday
start, end = [x.strip() for x in dates.split("-")]
start_date_list.append(start)
end_date_list.append(end)

players_table = single_soup.find(lambda tag: tag.name == "h2" and "Castaways" in tag.text).find_next("table")

names = []
links = []
names_and_links = players_table.findAll("td", {"align":"left"})
others = players_table.findAll("td", {"style":"text-align: left;"})
names_and_links.extend(others)
for each in names_and_links:
    found = each.find("a")
    names.append(found.text)
    links.append(found["href"])

assert len(names) == len(links)
players_names.append(names)
players_links.append(links)


season_46 = pd.DataFrame({
    "link":link, 
    "name":season_name,
    "season_no":season_no_list,
    "days":days_list,
    "start_date":start_date_list,
    "end_date":end_date_list,
    "num_castaways":castaways_num_list})

season_list = []
birthdate_list = []
hometown_list = []
occupation_list = []
player_description_list = []

for player_ind in range(len(players_links[0])):
    print(player_ind)
    season_list.append("Survivor: 46")
    
    # Go to the player's page for player-level information
    player = requests.get("https://survivor.fandom.com" + players_links[0][player_ind])

    player_soup = BeautifulSoup(player.text)

    # Easy data
    birthdate = player_soup.find("div", {"data-source":"birthdate"})
    if birthdate is not None:
        birthdate_list.append(birthdate.text.replace("\n", "").replace("Born", ""))
    else:
        birthdate_list.append(np.nan)

    hometown = player_soup.find("div", {"data-source":"hometown"}).text.replace("\n", "").replace("Hometown", "")
    hometown_list.append(hometown)
    
    occupation = player_soup.find("div", {"data-source":"occupation"}).text.replace("\n", "").replace("Occupation", "")
    occupation_list.append(occupation)

    # Description data
    all_text = ""

    for next_tag in player_soup.find(
        lambda tag: tag.name == "span" and tag.text == "Profile"
        ).find_next(lambda tag: tag.name == "span" and tag.text == "Profile").parent.next_siblings:
        if next_tag.name == "h2":
            break
        elif next_tag.text.startswith("Retrieved from "):
            continue
        elif next_tag.name == "table":
            player_seasons_list = next_tag.find("div", {"class":"wds-tabs__wrapper with-bottom-border"}).get_text("\n").split("\n")
            this_season = [index for index, item in enumerate(player_seasons_list) if item in "Survivor: 46"][0]
            
            if this_season == 0:
                all_text += next_tag.find("div", {"class":"wds-tab__content wds-is-current"}).text
            else:
                all_text += next_tag.findAll("div", {"class":"wds-tab__content"})[this_season-1].text
            
        else:
            all_text += next_tag.text

    player_description_list.append(all_text)
    
    

players_df = pd.DataFrame({
    "link":[j for i in players_links for j in i], 
    "name":[j for i in players_names for j in i],
    "season":season_list,
    "birthdate":birthdate_list,
    "hometown":hometown_list,
    "occupation":occupation_list,
    "player_description":player_description_list})


players_df = players_df.rename(columns={"link":"player_link", "name":"player_name", "season":"season_name"})
season_46 = season_46.rename(columns={"link":"season_link", "name":"season_name"})

all_df = pd.merge(players_df, season_46, how="left", on="season_name")

# Manually fix Venus's birthdate
all_df.loc[all_df["player_name"] == "Venus Vafa", "birthdate"] = "January 1, 1998"

def custom_slice(string, number):
    return string[:number]

# Fix birthday
all_df["birthdate"] = pd.Series(map(custom_slice, all_df["birthdate"], all_df["birthdate"].str.index(",")+6))


all_df["start_date"] = all_df["start_date"].str.replace(r'\[.*\]', '', regex=True)
all_df["end_date"] = all_df["end_date"].str.replace(r'\[.*\]', '', regex=True)
all_df["birthdate"] = all_df["birthdate"].str.replace(r'\[.*\]', '', regex=True)

all_df["start_date"] = pd.to_datetime(all_df["start_date"], format="%B %d, %Y")
all_df["end_date"] = pd.to_datetime(all_df["end_date"], format="%B %d, %Y")
all_df["birthdate"] = pd.to_datetime(all_df["birthdate"], format="%B %d, %Y")

all_df["age"] = (all_df["start_date"] - all_df["birthdate"]) / pd.Timedelta('365 days')



# Get embeddings
client = OpenAI(api_key = "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")

response = client.embeddings.create(
    input=all_df["occupation"][0],
    model="text-embedding-3-large"
)


occupation_embeddings = []
counter = 0
for each_occupation in all_df["occupation"]:
    if counter % 10 == 0:
        print(counter)
    response = client.embeddings.create(
        input=each_occupation,
        model="text-embedding-3-large",
        dimensions=256
    )

    occupation_embeddings.append(response.data[0].embedding)
    
    counter += 1



all_df["occupation_embedding"] = occupation_embeddings




description_embeddings = []
counter = 0
for each_description in all_df["player_description"]:
    if counter % 10 == 0:
        print(counter)
    response = client.embeddings.create(
        input=each_description,
        model="text-embedding-3-large",
        dimensions=256
    )

    description_embeddings.append(response.data[0].embedding)
    
    counter += 1

all_df["description_embedding"] = description_embeddings

all_df.to_csv("season_46.csv", index=False)
