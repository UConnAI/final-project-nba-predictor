import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from nba_api.stats.endpoints import commonallplayers, playergamelog
from nba_api.stats.static import players

def fetch_player_game_logs(player_id, season):
    game_logs = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    return game_logs

def prepare_data(df, num_lags):
    df[["REB", "AST", "PTS"]] = df[["REB", "AST", "PTS"]].apply(pd.to_numeric, errors='coerce')
    for col in ["REB", "AST", "PTS"]:
        for lag in range(1, num_lags + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df.dropna(inplace=True)
    return df

def predict_next_game_stats(player_name):
    #I use this to find the player_id using the player_name
    player_id = None
    for player in players.get_players():
        if player['full_name'] == player_name:
            player_id = player['id']
            break

    if player_id is None:
        print(f"Player {player_name} not found.")
        return

    #Here I fetch game logs for the specified player
    season = 2020
    game_logs = fetch_player_game_logs(player_id, season)

    #Prepare the data by creating lag features
    num_lags = 3
    prepared_data = prepare_data(game_logs, num_lags)

    for stat in ["PTS", "REB", "AST"]:
        #This for preprocessing
        X = prepared_data[[f"{stat}_lag{i}" for i in range(1, num_lags + 1)]]
        y = prepared_data[stat]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #And this is for model training
        model = LinearRegression()
        model.fit(X_train, y_train)

        #Prepare the input for the next game prediction
        next_game_input = pd.DataFrame(
            [prepared_data.iloc[-1][[f"{stat}_lag{i}" for i in range(1, num_lags + 1)]].values],
            columns=[f"{stat}_lag{i}" for i in range(1, num_lags + 1)])

        #stats for the next game
        next_game_stat = model.predict(next_game_input)[0]
        print(f"Raw predicted {stat}: {next_game_stat}")

        #this of course rounds the predicted stats to the nearest whole numbers
        next_game_stat = round(next_game_stat)

        #Finally we the final prediction
        print(f"Prediction for {player_name} in the next game: {next_game_stat} {stat}")

if __name__ == "__main__": 
    player_name = input("Enter the player's name: ")
    predict_next_game_stats(player_name)
