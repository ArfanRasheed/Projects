import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import difflib


def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['Name','Platform','Year','Genre','Publisher','NA Sales','EU Sales',
                  'JP Sales','Other Sales','Global Sales','Critic Score','Critic Count',
                  'User Score','User Count','Developer','Rating']
    df = df.drop(['Other Sales','Critic Count','User Count','User Score','Developer'], axis=1)
    df = df.replace('tbd', np.nan).dropna()

    for col in ['NA Sales','EU Sales','JP Sales','Year','Global Sales','Critic Score']:
        df[col] = df[col].astype('float')

    def bucket(series, edges):
        out = []
        for num in series:
            for i,(low,high) in enumerate(zip(edges, edges[1:]), start=1):
                if low <= num < high:
                    out.append(i)
                    break
            else:
                out.append(len(edges)-1)
        return out

    df['NA Sales'] = bucket(df['NA Sales'], [0,0.05,0.13,0.35,1e9])
    df['EU Sales'] = bucket(df['EU Sales'], [0,0.02,0.06,0.21,1e9])
    df['JP Sales'] = bucket(df['JP Sales'], [0,0.01,1e9])
    df['Global Sales'] = bucket(df['Global Sales'], [0,0.1,0.25,0.66,1e9])

    plat_map = {'Wii':1,'NES':1,'GB':1,'DS':1,'GBA':1,'3DS':1,'WiiU':1,'GC':1,
                'PS3':2,'PS2':2,'PS4':2,'PS':2,'PSP':2,'PSV':2,
                'X360':3,'XB':3,'XOne':3}
    df['Platform'] = df['Platform'].map(plat_map).fillna(0).astype(int)

    genre_map = {'Sports':1,'Platform':2,'Racing':3,'Role-Playing':4,'Puzzle':5,'Misc':6,
                 'Shooter':7,'Simulation':8,'Action':9,'Fighting':10,'Adventure':11,'Strategy':12}
    df['Genre'] = df['Genre'].map(genre_map).fillna(6).astype(int)

    df['Critic Score'] = pd.qcut(df['Critic Score'], q=10, labels=list(range(1,11))).astype(int)

    year_inds = []
    for num in df['Year']:
        if num >= 1985 and num < 2004:
            year_inds.append(1)
        elif num >= 2004 and num < 2007:
            year_inds.append(2)
        elif num >= 2007 and num < 2010:
            year_inds.append(3)
        elif num >= 2010:
            year_inds.append(4)
        else:
            year_inds.append(0)
    df['Year'] = year_inds

    return df


def data_info(df):
    print("Data shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head(3))
    print(df.describe().T[['count','mean','std']])


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
        z = self.encoder(x)
        return z, self.decoder(z)


def train_encoder(X, embedding_dim=16, epochs=30, batch_size=128, lr=1e-3,
                  val_frac=0.1, patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GameAutoEncoder(X.shape[1], embedding_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X, dtype=torch.float32)
    n = len(Xt)
    val_n = max(1, int(n * val_frac))
    idx = torch.randperm(n)
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt[train_idx]),
                                               batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt[val_idx]),
                                             batch_size=batch_size)

    best_val = float('inf')
    no_improve = 0
    best_state = None
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            _, recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= max(1, len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                _, recon = model(batch)
                val_loss += loss_fn(recon, batch).item() * batch.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}")

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model.encoder


def recommend_games(df, encoder, scaler, liked_game_names, top_k=5, allow_partial=True):
    feature_cols = ['Global Sales','Year', 'Genre', 'Platform']
    game_features = scaler.transform(df[feature_cols].values)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    game_tensors = torch.tensor(game_features, dtype=torch.float32).to(device)

    with torch.no_grad():
        embeddings = encoder(game_tensors).cpu().numpy()

    names_lower = df['Name'].str.lower()
    query_lower = [s.lower() for s in liked_game_names]
    liked_mask = names_lower.isin(query_lower)

    if not liked_mask.any() and allow_partial:
        matched = set()
        for q in query_lower:
            close = difflib.get_close_matches(q, names_lower.tolist(), n=3, cutoff=0.6)
            for c in close:
                matched.add(c)
        if matched:
            liked_mask = names_lower.isin(list(matched))

    liked_idx = np.where(liked_mask)[0]
    if len(liked_idx) == 0:
        return []

    mean_embed = embeddings[liked_idx].mean(axis=0, keepdims=True)
    sims = cosine_similarity(embeddings, mean_embed).reshape(-1)
    sims[liked_idx] = -np.inf
    rec_idx = np.argsort(-sims)[:top_k]
    return df.iloc[rec_idx]['Name'].tolist()


def train_classifiers(df):
    """Simple demo training similar to notebook portion."""
    feature_cols = ['Global Sales','Year','Genre','Platform']
    X = df[feature_cols]
    y = df['Critic Score']
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.15,
                                              shuffle=True, random_state=20)
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(x_tr, y_tr)
    print("DT test acc", accuracy_score(y_te, clf.predict(x_te)))
    lr = LogisticRegression(solver='liblinear', max_iter=1000)
    lr.fit(x_tr, y_tr)
    print("LR test acc", accuracy_score(y_te, lr.predict(x_te)))
    return clf, lr


def get_recommendations_from_titles(liked_titles, csv_path, top_k=5, epochs=30):
    df = preprocess_data(csv_path)
    data_info(df)
    feature_cols = ['Global Sales','Year','Genre','Platform']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    encoder = train_encoder(X, embedding_dim=16, epochs=epochs)
    recs = recommend_games(df, encoder, scaler, liked_titles, top_k=top_k)
    return recs

if __name__ == "__main__":
    csv_path = r'c:\Users\Arfan\Documents\Projects\Machine Learning Project\videogame_sales.csv'
    df = preprocess_data(csv_path)
    data_info(df)
    train_classifiers(df)
    likes = ['Wii Sports','Super Mario Bros.']
    print("Recs:", get_recommendations_from_titles(likes, csv_path, top_k=5, epochs=20))