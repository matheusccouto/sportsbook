# %% [markdown]
# # Football Betting using FiveThirtyEight's Soccer Power Index
# ## Goal
# Backtest a strategy of using the FiveThirtyEight's Soccer Power Index for betting.

# %% [markdown]
# ## Imports

# %%
import concurrent.futures
import io
import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "data")


# %% [markdown]
# ## FiveThirtyEight
# Download Soccer Power Index dataset.

# %%
SPI_SOCCER_URL = "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"

content = requests.get(SPI_SOCCER_URL).content
df_spi = pd.read_csv(io.StringIO(content.decode()))
df_spi["date"] = pd.to_datetime(df_spi["date"], format="%Y-%m-%d")

assert df_spi["date"].isna().sum() == 0

df_spi.sample(5)


# %% [markdown]
# ## Football-Data.co.uk
# [football-data.co.uk](https://www.football-data.co.uk) is a website that provides historical betting odds for many soccer leagues.

# %%
FOOTBALL_DATA_MAIN_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
LEAGUES = {
    "E0": ["Barclays Premier League"],
    "E1": ["English League Championship"],
    "E2": ["English League One"],
    "E3": ["English League Two"],
    "SC0": ["Scottish Premiership"],
    "D1": ["German Bundesliga"],
    "D2": ["German 2. Bundesliga"],
    "I1": ["Italy Serie A"],
    "I2": ["Italy Serie B"],
    "SP1": ["Spanish Primera Division"],
    "SP2": ["Spanish Segunda Division"],
    "F1": ["French Ligue 1"],
    "F2": ["French Ligue 2"],
    "N1": ["Dutch Eredivisie"],
    "B1": ["Belgian Jupiler League"],
    "P1": ["Portuguese Liga"],
    "T1": ["Turkish Turkcell Super Lig"],
    "G1": ["Greek Super League"],
}

FOOTBALL_DATA_OTHER_URL = "https://www.football-data.co.uk/new/{league}.csv"
OTHER_LEAGUES = {
    "ARG": ["Argentina Primera Division"],
    "AUT": ["Austrian T-Mobile Bundesliga"],
    "BRA": ["Brasileiro Série A"],
    "CHN": ["Chinese Super League"],
    "DNK": ["Danish SAS-Ligaen"],
    "JPN": ["Japanese J League"],
    "MEX": ["Mexican Primera Division Torneo Apertura", "Mexican Primera Division Torneo Clausura"],
    "NOR": ["Norwegian Tippeligaen"],
    "RUS": ["Russian Premier Liga"],
    "SWE": ["Swedish Allsvenskan"],
    "SWZ": ["Swiss Raiffeisen Super League"]
}


def url_to_pandas(url):
    """Download URL content to a pandas dataframe."""
    content = requests.get(url).content
    data = pd.read_csv(io.StringIO(content.decode(encoding="latin1")))
    data = data.dropna(how="all", axis=0)
    data = data.dropna(how="all", axis=1)
    data["URL"] = url
    return data


def get_football_data_main(year, league):
    """Get football data."""
    season = str(year - 1)[-2:] + str(year)[-2:]
    url = FOOTBALL_DATA_MAIN_URL.format(season=season, league=league)
    data = url_to_pandas(url)
    data["Season"] = season
    return data


def get_football_data_other(league):
    """Get football data."""
    url = FOOTBALL_DATA_OTHER_URL.format(league=league)
    data = url_to_pandas(url)
    data["Div"] = league
    return data



with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(get_football_data_main, year=year, league=league)
        for league in LEAGUES.keys()
        for year in range(2023, 2015, -1)
    ]
    results_main = [future.result() for future in concurrent.futures.as_completed(futures)]

    futures = [
        executor.submit(get_football_data_other, league=league)
        for league in OTHER_LEAGUES.keys()
    ]
    results_other = [future.result() for future in concurrent.futures.as_completed(futures)]

df_bet_main = pd.concat(results_main)
df_bet_other = pd.concat(results_other)


# %%
df_bet_other = df_bet_other.rename(
    columns={
        "Home": "HomeTeam",
        "Away": "AwayTeam",
        "HG": "FTHG",
        "AG": "FTAG",
        "Res": "FTR",
        "PH": "PSH",
        "PD": "PSD",
        "PA": "PSA",
    }
)


# %%
df_bet = pd.concat([df_bet_main, df_bet_other])


# %% [markdown]
# Dates comes in differents formats depending on the year.

# %%
date1 = pd.to_datetime(df_bet["Date"], format="%d/%m/%Y", errors="coerce")
date2 = pd.to_datetime(df_bet["Date"], format="%d/%m/%y", errors="coerce")
df_bet["Date"] = date1.fillna(date2)

assert df_bet["Date"].isna().sum() == 0

df_bet.sample(5)


# %% [markdown]
# Each data source uses different ways of writing the same clubs names. I will use a dict to make names the same.

# %%
# from thefuzz import fuzz
# import networkx as nx

# # Generate empty names dict

# frames = [df_spi["team1"], df_spi["team2"], df_bet["HomeTeam"], df_bet["AwayTeam"]]
# names = pd.concat(frames).drop_duplicates()

# pairs = [
#     (name, other_name)
#     for name in names
#     for other_name in names
#     if fuzz.partial_ratio(name, other_name) > 90
# ]

# graph = nx.Graph()
# graph.add_edges_from(pairs)

# clusters = [list(cluster) for cluster in nx.connected_components(graph)]
# clusters = {cluster[0]: cluster for cluster in clusters}
# (
#     pd.Series(clusters)
#     .sort_index()
#     .to_json(os.path.join(DATA_DIR, "names.json"), force_ascii=False)
# )


# %%
with open(os.path.join(DATA_DIR, "names.json"), encoding="utf-8") as file:
    names_dict = json.load(file)


names_dict = {
    name.strip(): i.strip()
    for i, name_list in names_dict.items()
    for name in name_list
}


def clean(series, translate_dict):
    """Clean text in pandas series."""
    return (
        series
        .str.strip()
        .apply(lambda x: translate_dict[x] if x in translate_dict else x)
    )


df_spi["home"] = clean(df_spi["team1"], names_dict)
df_spi["away"] = clean(df_spi["team2"], names_dict)
df_bet["home"] = clean(df_bet["HomeTeam"], names_dict)
df_bet["away"] = clean(df_bet["AwayTeam"], names_dict)


# %%
df_bet_before = df_bet.copy()
df_bet_after = df_bet.copy()

df_bet_before["Date"] = df_bet_before["Date"] - pd.Timedelta(days=1)
df_bet_after["Date"] = df_bet_before["Date"] + pd.Timedelta(days=1)

df_bet_expanded = pd.concat((df_bet, df_bet_before, df_bet_after))

# %%
leagues = [league for leagues in LEAGUES.values() for league in leagues]
other_leagues = [league for leagues in OTHER_LEAGUES.values() for league in leagues]

df_spi = (
    df_spi
    .query(f"league in {leagues + other_leagues}")
    .query(f"date < '{datetime.today().date()}'")
)

# %%
df_bet_expanded["dt"] = df_bet_expanded["Date"].dt.date
df_spi["dt"] = df_spi["date"].dt.date
df = df_spi.convert_dtypes().merge(df_bet_expanded.convert_dtypes(), how="left", on=["dt", "home", "away"], indicator=True)

# %%
1/0

# %%
df_bet_expanded.query("Date == '2018-07-19' and Div == 'BRA'").sort_values("Date")

# %%
df.query("_merge == 'left_only'")

# %%
1/0

# %% [markdown]
# There are only a few clubs left that are unmatch. Maybe one dataset has more games than the other.
# 
# Now that names are fixed, it is able to be merged.

# %%
df_bet = df_bet.rename(
    {"Date": "date", "HomeTeam_clean": "team1_clean", "AwayTeam_clean": "team2_clean"}, axis=1
)

# df_bet_minus = df_bet.copy()
# df_bet_minus["date"] = df_bet_minus["date"] - pd.Timedelta(days=1)

# df_bet_plus = df_bet.copy()
# df_bet_plus["date"] = df_bet_plus["date"] + pd.Timedelta(days=1)

# df_bet = df_bet.append(df_bet_minus)
# df_bet = df_bet.append(df_bet_plus)


# %%
df = df_spi.merge(df_bet, how="outer", on=["date", "team1_clean", "team2_clean"], validate="1:1", indicator=True)

print(df_spi.shape[0] - df.shape[0])

# %%
df.query("_merge == 'left_only'")[["date", "league", "team1_clean", "team2_clean", "_merge"]]

# %% [markdown]
# Add each results point of view to the dataset.

# %%
df["win"] = df["score1"] > df["score2"]
df["draw"] = df["score1"] == df["score2"]
df["loss"] = df["score1"] < df["score2"]

df_inv = df.copy()

df_inv["team2"], df_inv["team1"] = df["team1"], df["team2"]
df_inv["spi2"], df_inv["spi1"] = df["spi1"], df["spi2"]
df_inv["prob2"], df_inv["prob1"] = df["prob1"], df["prob2"]
df_inv["proj_score2"], df_inv["proj_score1"] = df["proj_score1"], df["proj_score2"]
df_inv["importance2"], df_inv["importance1"] = df["importance1"], df["importance2"]
df_inv["score2"], df_inv["score1"] = df["score1"], df["score2"]
df_inv["xg2"], df_inv["xg1"] = df["xg1"], df["xg2"]
df_inv["nsxg2"], df_inv["nsxg1"] = df["nsxg1"], df["nsxg2"]
df_inv["adj_score2"], df_inv["adj_score1"] = df["adj_score1"], df["adj_score2"]
df_inv["B365A"], df_inv["B365H"] = df["B365H"], df["B365A"]
df_inv["MaxA"], df_inv["MaxH"] = df["MaxH"], df["MaxA"]
df_inv["AvgA"], df_inv["AvgH"] = df["AvgH"], df["AvgA"]
df_inv["loss"], df_inv["win"] = df["win"], df["loss"]

df_draw = df.copy()
df_draw["team1"] = "draw"
df_draw["team2"] = np.nan
df_draw["prob1"] = df["probtie"]
df_draw["B365H"] = df["B365D"]
df_draw["MaxH"] = df["MaxD"]
df_draw["AvgH"] = df["AvgD"]
df_draw["win"] = df["draw"]

df = df.append(df_inv).reset_index(drop=True)
df = df.append(df_draw).reset_index(drop=True)

print(df.shape)

df.sample(5)


# %% [markdown]
# ## Results
# ### ROI Lines

# %%
def roi_lines(data, odds_col):
    """Calculate ROI."""
    data["ev"] = data["prob1"] * (data[odds_col] - 1) - (1 - data["prob1"])
    data["bet"] = data["ev"] > 0
    data["balance"] = data["bet"].astype(int) * (
        data["win"].astype(int) * data[odds_col] - 1
    )

    return data["balance"].sum() / data["bet"].sum()


print(f"Avg ROI = {roi_lines(df, 'AvgH') * 100:.2g}%")
print(f"B365 ROI = {roi_lines(df, 'B365H') * 100:.2g}%")
print(f"Max ROI = {roi_lines(df, 'MaxH') * 100:.2g}%")


# %% [markdown]
# ### ROI Over/Under

# %%
df[">2.5"] = df["score1"] + df["score2"] > 2.5
df["<2.5"] = df["score1"] + df["score2"] < 2.5


def roi_over_under(data, odds_col):
    """Calculate ROI."""
    pd.options.mode.chained_assignment = None
    data = data.dropna(subset=[f"{odds_col}<2.5", f"{odds_col}>2.5", "team2"])
    data["bet_over"] = data["x>2.5"].astype(int)
    data["bet_under"] = data["x<2.5"].astype(int)
    data["balance_over"] = data["bet_over"] * (
        data[">2.5"].astype(int) * df[f"{odds_col}>2.5"] - 1
    )
    data["balance_under"] = data["bet_under"] * (
        data["<2.5"].astype(int) * df[f"{odds_col}<2.5"] - 1
    )
    data["balance"] = data["balance_over"] + data["balance_under"]
    return data["balance"].sum() / data["bet"].sum()


df["x>2.5"] = df["proj_score1"] + df["proj_score1"] > 2.5
df["x<2.5"] = df["proj_score1"] + df["proj_score1"] < 2.5

print(f"Avg ROI = {roi_over_under(df, 'Avg') * 100:.2g}%")
print(f"B365 ROI = {roi_over_under(df, 'B365') * 100:.2g}%")
print(f"Max ROI = {roi_over_under(df, 'Max') * 100:.2g}%")


# %% [markdown]
# ## Conclusion
# ### Lines
# This strategy would lose money against the average betting site. However, it is able to have a small margin agains some specific websites that offers good odds.
# 
# ### Over/Under
# The over/under strategy is not profitable, even considering the best odds available.


