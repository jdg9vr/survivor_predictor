import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
# import gender_guesser.detector as gender
from openai import OpenAI

def scrape_all_seasons():
    """
    This function scrapes the webpage with all Survivor seasons (https://survivor.fandom.com/wiki/Survivor_(U.S.)#Seasons).
    Its purpose is to get the names ad links for each Survivor season
    Input: None
    Output: (survivors_links, survivor_names): ([str], [str]) = tuple of lists of each survivor season's link and name
    """

    # Get all seasons
    seasons = requests.get("https://survivor.fandom.com/wiki/Survivor_(U.S.)#Seasons")

    # Pull out the links to each season
    seasons_soup = BeautifulSoup(seasons.text)
    table = seasons_soup.find("table", {"class":"wikitable", 
                                        "style":"margin:auto; text-align:center; font-size:8pt; line-height:18px;"})


    filtered = table.findAll("a", title=re.compile("Survivor:*"))

    survivors_links = []
    season_names = []
    count = 0
    for f in filtered:
        if count % 2 == 0:
            survivors_links.append(f["href"])
            season_names.append(f["title"])
        count += 1

    assert len(survivors_links) == len(season_names)
    
    return survivors_links, season_names

def scrape_each_season(survivors_links, season_names):
    """
    Cycle through Survivor season Wiki links, get the winner, season number, # of days, dates, # of castaways, and 
    a list for each season of each player and their Survivor link.
    Input: survivors_links: [str] = All links to each Survivor season.
           season_names: [str] = All names to each Survivor season.
    Output: seasons_df: pd.DataFrame = DataFrame of each season's details
            players_names: [[str]] = List of lists of players for each seasons (first list = season 1, second list = season 2)
            players_links: [[str]] = List of lists of player links for each season
    """

    winners_list = []
    season_no_list = []
    days_list = []
    dates_list = []
    start_date_list = []
    end_date_list = []
    castaways_num_list = []
    players_names = []
    players_links = []
    for season in range(len(survivors_links)):
        print(season)
        link = survivors_links[season]

        # Get all the names and links for each player that season
        single_season = requests.get("https://survivor.fandom.com" + link)

        single_soup = BeautifulSoup(single_season.text)

        try:
            single_winner = single_soup.find("div", {"data-source":"winner"}).text.replace("\n", "").replace("Winner", "")
            winners_list.append(single_winner)
        except:
            print(f"Season #{season+1}, {season_names[season]} has no winner")
            single_winner = None
            winners_list.append(np.nan)

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


    seasons_df = pd.DataFrame({
        "link":survivors_links, 
        "name":season_names,
        "season_no":season_no_list,
        "days":days_list,
        "start_date":start_date_list,
        "end_date":end_date_list,
        "num_castaways":castaways_num_list,
        "winner":winners_list})
    
    seasons_df = seasons_df.rename(columns={"link":"season_link", "name":"season_name"})
    
    return seasons_df, players_names, players_links

def get_players(players_names, players_links, season_names):
    """
    Go to each player's webpage and fetch their name, birthday, hometown, occupation, and description.
    Input: players_names: [[str]] = List of lists of players for each seasons (first list = season 1, second list = season 2)
           players_links: [[str]] = List of lists of player links for each season
           season_names: [str] = All names to each Survivor season
    Output: players_df: pd.DataFrame = DataFrame of each players's details
    """

    season_list = []
    birthdate_list = []
    hometown_list = []
    occupation_list = []
    player_description_list = []
    for season in range(len(season_names)):
        for player_ind in range(len(players_links[season])):
            print(season, player_ind)
            season_list.append(season_names[season])
            
            # Go to the player's page for player-level information
            player = requests.get("https://survivor.fandom.com" + players_links[season][player_ind])

            player_soup = BeautifulSoup(player.text)

            # Easy data - check if it exists first
            birthdate_first = player_soup.find("div", {"data-source":"birthdate"})
            birthdate = birthdate_first.text.replace("\n", "").replace("Born", "") if birthdate_first is not None else None
            if birthdate is None:
                print(f"Unknown birthdate: Season #{season}, {season_names[season]}; Player #{player_ind}, {players_names[season][player_ind]}")
            birthdate_list.append(birthdate)
            
            hometown_first = player_soup.find("div", {"data-source":"hometown"})
            hometown = hometown_first.text.replace("\n", "").replace("Hometown", "") if hometown_first is not None else None
            if hometown is None:
                print(f"Unknown hometown: Season #{season}, {season_names[season]}; Player #{player_ind}, {players_names[season][player_ind]}")
            hometown_list.append(hometown)
            
            occupation_first = player_soup.find("div", {"data-source":"occupation"})
            occupation = occupation_first.text.replace("\n", "").replace("Occupation", "") if occupation_first is not None else None
            if occupation is None:
                print(f"Unknown occupation: Season #{season}, {season_names[season]}; Player #{player_ind}, {players_names[season][player_ind]}")
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
                    # For players who have played more than once - find the correct season
                    player_seasons_list = next_tag.find("div", {"class":"wds-tabs__wrapper with-bottom-border"}).get_text("\n").split("\n")
                    this_season = [index for index, item in enumerate(player_seasons_list) if item in season_names[season]][0]
                    
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
    
    return players_df


def merge_and_clean(players_df, seasons_df):
    """
    Combine ad clean up the dataset, including creating age, asserting 
    Input: players_df: pd.DataFrame = DataFrame of each players's details
           seasons_df: pd.DataFrame = DataFrame of each season's details
    Output: all_df: pd.DataFrame = combined and cleaned DataFrame (each row is a player)
    """
    all_df = pd.merge(players_df, seasons_df, how="left", on="season_name")


    def custom_slice(string, number):
        return string[:number]

    # Fix birthday
    all_df["birthdate"] = pd.Series(map(custom_slice, all_df["birthdate"], all_df["birthdate"].str.index(",")+6))

    # Get winner or non-winner binary to predict on
    all_df["is_winner"] = np.where(all_df["player_name"] == all_df["winner"], "Winner", "Non-Winner")
    all_df["is_winner"] = np.where(all_df["winner"].isna(), np.nan, all_df["is_winner"])

    # Test that all the players for each season are in the table
    test_df = all_df.groupby("season_name").agg({"player_name":"count", "num_castaways":"first"})
    test_df["num_castaways"] = test_df["num_castaways"].astype(int)
    assert (test_df["player_name"] == test_df["num_castaways"]).sum() == len(seasons_df)

    all_df["start_date"] = all_df["start_date"].str.replace(r'\[.*\]', '', regex=True)
    all_df["end_date"] = all_df["end_date"].str.replace(r'\[.*\]', '', regex=True)
    all_df["birthdate"] = all_df["birthdate"].str.replace(r'\[.*\]', '', regex=True)

    all_df["start_date"] = pd.to_datetime(all_df["start_date"], format="%B %d, %Y")
    all_df["end_date"] = pd.to_datetime(all_df["end_date"], format="%B %d, %Y")
    all_df["birthdate"] = pd.to_datetime(all_df["birthdate"], format="%B %d, %Y")

    all_df["age"] = (all_df["start_date"] - all_df["birthdate"]) / pd.Timedelta('365 days')

    # d = gender.Detector()

    # all_df["player_name"].str.split(" ").str[0].map(d.get_gender).value_counts()

    # d.get_gender()
    
    return all_df



def get_occupation_embeddings(occupations, client):
    """
    Embed the text for each player's occupation with OpenAI Embedding model/
    Input: occupations: pd.Series = Series of all the occupations as strings.
           client: OpenAI() = Instantiated OpenAI client for creating embeddings
    Output: occupation_embeddings: [[float]] = A list of lists of embeddings for the occupations of the players.
    """
    occupation_embeddings = []
    counter = 0
    for each_occupation in occupations:
        if counter % 10 == 0:
            print(counter)
        response = client.embeddings.create(
            input=each_occupation,
            model="text-embedding-3-large",
            dimensions=256
        )

        occupation_embeddings.append(response.data[0].embedding)
        
        counter += 1
    
    return occupation_embeddings

def get_description_embeddings(descriptions, client):
    """
    Embed the text for each player's description with OpenAI Embedding model/
    Input: all_df: pd.Series = Series of all the descriptions for each player as strings
           client: OpenAI() = Instantiated OpenAI client for creating embeddings
    Output: descriptiontion_embeddings: [[float]] = A list of lists of embeddings for the descriptions of the players.
    """
    description_embeddings = []
    counter = 0
    for each_description in descriptions:
        if counter % 10 == 0:
            print(counter)
        response = client.embeddings.create(
            input=each_description,
            model="text-embedding-3-large",
            dimensions=256
        )

        description_embeddings.append(response.data[0].embedding)
        
        counter += 1
    
    return description_embeddings

