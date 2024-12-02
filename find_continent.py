import pandas as pd
import json

# On cherche à trouver le continent de chaque pays de la liste du csv
# Création d'un country.json depuis  https://restcountries.com


# Chargement du CSV
df = pd.read_csv('most-dangerous-countries-for-women-2024.csv')

# Chargement du Json
# Charger le JSON (mettez le chemin de votre fichier JSON)
with open('countries.json', 'r') as f:
    countries_data = json.load(f)


# Dictionnaire pour stocker le continent de chaque pays
country_continents = {}

# Pour chaque pays dans le CSV, chercher le continent correspondant dans le JSON
for country in df['country']:
    for country_data in countries_data:
        if country_data["name"]["common"] == country:
            country_continents[country] = country_data["region"]
            break  # Arrêter la recherche dès qu'on trouve le pays

# Ajouter la colonne 'continent' au DataFrame CSV en utilisant le dictionnaire créé
df['continent'] = df['country'].map(country_continents)


df.to_csv('most-dangerous-countries-for-women-2024.csv', index=False)
