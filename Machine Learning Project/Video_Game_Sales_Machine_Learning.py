# Video_Game_Sales_Machine_Learning.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['Name','Platform','Year','Genre','Publisher','NA Sales','EU Sales', 'JP Sales','Other Sales','Global Sales','Critic Score','Critic Count','User Score','User Count' ,'Developer', 'Rating']
    df = df.drop(['Other Sales', 'Critic Count','User Count','User Score','Developer'], axis = 1)
    df = df.replace('tbd',np.nan).dropna()
    df['NA Sales'] = df['NA Sales'].astype('float')
    df['EU Sales'] = df['EU Sales'].astype('float')
    df['JP Sales'] = df['JP Sales'].astype('float')
    df['Year'] = df['Year'].astype('float')
    df['Global Sales'] = df['Global Sales'].astype('float')
    df['Critic Score'] = df['Critic Score'].astype('float')

    # Quantile binning for sales columns (better class balance)
    for col in ['NA Sales', 'EU Sales', 'JP Sales', 'Global Sales']:
        # Find number of unique bins first
        try:
            bins = pd.qcut(df[col], q=4, retbins=True, duplicates='drop')[1]
            n_bins = len(bins) - 1
            labels = list(range(1, n_bins+1))
            df[col] = pd.qcut(df[col], q=n_bins, labels=labels, duplicates='drop').astype(int)
        except Exception as e:
            print(f"Could not bin column {col}: {e}")

    # Encode Platform
    plat_map = {'Wii':1, 'NES':1, 'GB':1, 'DS':1, 'GBA':1, '3DS':1, 'WiiU':1, 'GC':1,
                'PS3':2, 'PS2':2, 'PS4':2, 'PS':2, 'PSP':2, 'PSV':2,
                'X360':3, 'XB':3, 'XOne':3}
    df['Platform'] = df['Platform'].map(plat_map).fillna(0).astype(int)

    # Encode Genre
    genre_map = {'Sports':1, 'Platform':2, 'Racing':3, 'Role-Playing':4, 'Puzzle':5, 'Misc':6,
                 'Shooter':7, 'Simulation':8, 'Action':9, 'Fighting':10, 'Adventure':11, 'Strategy':12}
    df['Genre'] = df['Genre'].map(genre_map).fillna(6).astype(int)

    # Encode Publisher (label encoding)
    df['Publisher'] = df['Publisher'].astype('category').cat.codes

    # Quantile binning for Critic Score
    df['Critic Score'] = pd.qcut(df['Critic Score'], q=5, labels=[1,2,3,4,5]).astype(int)

    # Bin Year
    year_bins = [0,1985,2004,2007,2010,np.inf]
    year_labels = [0,1,2,3,4]
    df['Year'] = pd.cut(df['Year'], bins=year_bins, labels=year_labels, include_lowest=True).astype(int)

    return df

def train_and_evaluate(df, plot=True):
    # Use more features for better accuracy
    feature_cols = ['Global Sales','NA Sales','EU Sales','JP Sales','Year', 'Genre', 'Platform', 'Publisher']
    X = df[feature_cols]
    y = df['Critic Score']
    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.15, random_state=20, stratify=y)

    # Decision Tree with limited depth to reduce overfitting
    clf_dt = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    clf_dt.fit(x_train, y_train)
    y_train_pred = clf_dt.predict(x_train)
    y_test_pred = clf_dt.predict(x_test)
    a_dt_train = accuracy_score(y_train, y_train_pred)
    a_dt_test = accuracy_score(y_test, y_test_pred)
    print("Decision Tree (max_depth=5) Training Accuracy:", a_dt_train)
    print("Decision Tree (max_depth=5) Test Accuracy:", a_dt_test)

    # Random Forest (less prone to overfitting)
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf_rf.fit(x_train, y_train)
    y_rf_train = clf_rf.predict(x_train)
    y_rf_test = clf_rf.predict(x_test)
    print("Random Forest (max_depth=5) Training Accuracy:", accuracy_score(y_train, y_rf_train))
    print("Random Forest (max_depth=5) Test Accuracy:", accuracy_score(y_test, y_rf_test))

    # Logistic Regression with regularization
    model = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000, C=0.5)
    model.fit(x_train, y_train)
    print("Logistic Regression Test Score:", model.score(x_test, y_test))
    print(classification_report(y_test, model.predict(x_test)))
    print(confusion_matrix(y_test, model.predict(x_test)))

    return clf_dt, clf_rf, model, x_test, y_test

# PyTorch AutoEncoder Model
class GameAutoEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        embedding = self.encoder(x)
        recon = self.decoder(embedding)
        return embedding, recon

def train_encoder(X, embedding_dim=16, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GameAutoEncoder(X.shape[1], embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        embedding, recon = model(X_tensor)
        loss = criterion(recon, X_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return model.encoder  # Return only the encoder part for recommendations

def recommend_games(df, model, scaler, liked_game_names, top_k=5):
    feature_cols = ['Global Sales','Year', 'Genre', 'Platform']
    game_features = scaler.transform(df[feature_cols].values)
    game_tensors = torch.tensor(game_features, dtype=torch.float32)
    # Recommend games based on liked games using learned embeddings
    with torch.no_grad():
        embeddings = model(game_tensors).numpy()  # Get embeddings for all games
    # Find indices of liked games
    liked_idx = df[df['Name'].isin(liked_game_names)].index
    liked_embeds = embeddings[liked_idx]  # Embeddings for liked games
    mean_embed = liked_embeds.mean(axis=0)  # Average embedding
    # Compute cosine similarity between all games and the mean embedding
    sims = embeddings @ mean_embed / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(mean_embed) + 1e-8)
    # Exclude liked games from recommendations
    rec_idx = np.argsort(-sims)
    rec_idx = [i for i in rec_idx if i not in liked_idx][:top_k]
    return df.iloc[rec_idx][['Name', 'Platform', 'Genre']]

# Run basic tests for the classical ML models
# Checks if accuracy is above a threshold

def run_tests():
    df = preprocess_data(r'c:\Users\Arfan\Documents\Projects\Machine Learning Project\videogame_sales.csv')
    clf_dt, clf_rf, model, x_test, y_test = train_and_evaluate(df, plot=False)
    # Example test: check if accuracy is above a threshold
    dt_acc = accuracy_score(y_test, clf_dt.predict(x_test))
    rf_acc = accuracy_score(y_test, clf_rf.predict(x_test))
    lr_acc = accuracy_score(y_test, model.predict(x_test))
    print(f"Tested Decision Tree accuracy: {dt_acc:.2f}")
    print(f"Tested Random Forest accuracy: {rf_acc:.2f}")
    print(f"Tested Logistic Regression accuracy: {lr_acc:.2f}")
    assert dt_acc > 0.2, "Decision Tree accuracy too low!"
    assert rf_acc > 0.2, "Random Forest accuracy too low!"
    assert lr_acc > 0.2, "Logistic Regression accuracy too low!"

if __name__ == "__main__":
    # Preprocess the data
    df = preprocess_data(r'c:\Users\Arfan\Documents\Projects\Machine Learning Project\videogame_sales.csv')
    # Train and evaluate classical ML models (Decision Tree, Logistic Regression)
    train_and_evaluate(df)
    # --- Recommendation system demo ---
    df = preprocess_data(r'c:\Users\Arfan\Documents\Projects\Machine Learning Project\videogame_sales.csv')
    feature_cols = ['Global Sales','Year', 'Genre', 'Platform']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    # Train the PyTorch encoder (autoencoder-style)
    encoder = train_encoder(X, embedding_dim=16, epochs=30)
    # List of liked games for recommendation
    liked_games = ['Super Mario Bros.', 'Call of Duty: Black Ops', 'Wii Sports']  # Example
    # Get recommendations
    recs = recommend_games(df, encoder, scaler, liked_games, top_k=5)
    print("Recommended games:")
    print(recs)
    # Uncomment to run tests
    run_tests()