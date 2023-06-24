import datetime
import random

import pandas as pd
import unicodedata

FILE_PATH = "C:\\Users\\Hugo\\Desktop\\read_convert_csv\\Tweets_incendio_completed.csv"
PARISHES_PATH = "C:\\Users\\Hugo\\Desktop\\read_convert_csv\\Parishes.txt"
SAVE_PATH = "C:\\Users\\Hugo\\Desktop\\read_convert_csv\\october_fires.txt"

# these imports are needed for some geopandas tools
import matplotlib
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point

# make sure its a folder not file
GEO_FILE_PATH_PT = "C:\\Users\\Hugo\\Desktop\\read_convert_csv\\CAOP_Continente_2022"
GEO_FILE_PATH_MAD = "C:\\Users\\Hugo\\Desktop\\read_convert_csv\\CAOP_Madeira_2022"
GEO_FILE_PATH_AZO = "C:\\Users\\Hugo\\Desktop\\read_convert_csv\\CAOP_Azores_2022"

PORTUGAL_GDF = gpd.read_file(GEO_FILE_PATH_PT)
MADEIRA_GDF = gpd.read_file(GEO_FILE_PATH_MAD)
AZORES_GDF = gpd.read_file(GEO_FILE_PATH_AZO)


# useless now since geopandas handles this
DISTRICTS = [
    "aveiro",
    "beja",
    "braga",
    "bragança",
    "castelo branco",
    "coimbra",
    "evora",
    "faro",
    "guarda",
    "leiria",
    "lisboa",
    "portalegre",
    "porto",
    "santarem",
    "setubal",
    "viana do castelo",
    "vila real",
    "viseu"
]

HAZARDS = [
    # SIGNS
    "fumo",
    "fogo",
    "incêndio",
    "chamas",
    # FUELS
    "explosivo",
    "fertilizante",
    "quimico",
    "gas",
    "gasolina",
    "petroleo",
    # HOUSES & PEOPLE
    "urbana",
    "urbanização",
    "casa",
    "populacao",
    "hospital",
    # FIRE RELATED
    "ignicao",
    "combustivel",
    "propagacao",
    "queimada",
    "rescaldo",
    "incendio"
]


def generate_string():
    text_hazard = random.choice(HAZARDS)

    if random.random() < 0.2:
        text_hazard += f'-{random.choice(HAZARDS)}'

    return text_hazard


# useless now since geopandas handles this
def read_parishes_file(file_path):
    parishes = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for l in file:
            parish = l.strip().lower()
            if parish:
                # Replace special characters with their standard versions
                parish = unicodedata.normalize('NFKD', parish).encode('ASCII', 'ignore').decode('utf-8')
                parishes.append(parish)

    return parishes


PARISHES = read_parishes_file(PARISHES_PATH)


def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df


def process_excel_file(file_path):
    data = []
    df = read_csv_file(file_path)
    i = 0
    users = {}

    for _, row in df.iterrows():
        submission_id = i

        # Select an existing user or create a new unique user
        if len(users) > 0 and random.random() < 0.2:
            user_id, user_data = random.choice(list(users.items()))
            user_rating = user_data["rating"]
        else:
            user_id = i  # Create a new unique user
            user_rating = random.randint(1, 20)  # Assign a random rating between 1 and 20
            users[user_id] = {"rating": user_rating}  # Store the new user

        submission_date_str = row['Datetime']
        hour = int(row['Hour'])
        minutes = random.randint(0, 59)
        submission_date = datetime.datetime.strptime(submission_date_str, '%Y-%m-%d')
        submission_date = submission_date.replace(hour=hour, minute=minutes)
        submission_date = submission_date.strftime('%d/%m/%Y %H:%M')

        lat = float(row['Latitude'])
        lon = float(row['Longitude'])
        fire_verified = row['Fire']
        if fire_verified:
            fire_verified = 0
        else:
            fire_verified = 1
        smoke_verified = random.choice([0, 1])  # Randomly choose True or False for smoke_verified

        # ignore text
        text = row['Text']

        # create point
        pt = (lat, lon)

        # use geopandas to get point data
        district, parish = get_concelho_distrito(pt, PORTUGAL_GDF, MADEIRA_GDF, AZORES_GDF)

        text_district = district
        text_parish = parish
        text_keywords = ''

        if random.random() < 0.2:
            text_keywords = generate_string()

        data.append((submission_id, submission_date, user_id, user_rating, fire_verified, smoke_verified,
                     lat, lon, text_district, text_parish, text_keywords))
        i += 1

    return data


def save_data_to_txt(data, file_path):
    with open(file_path, 'w') as file:
        # Write the header
        header = "sub_id;date/time;user_id;user_rating;fire_verified;smoke_verified;lat;lon;text_district;text_loc;text_keywords\n"
        file.write(header)

        # Write the data rows
        for row in data:
            submission_id, submission_date, user_id, user_rating, fire_verified, smoke_verified, lat, lon, text_district, text_parish, text_keywords = row

            # Format the row as a string
            row_str = f"{submission_id};{submission_date};{user_id};{user_rating};{fire_verified};{smoke_verified};{lat};{lon};{text_district};{text_parish};{text_keywords}\n"

            # Write the row to the file
            file.write(row_str)

    print(f"Data saved to file: {file_path}")


def get_concelho_distrito(point, district_parish_gdf, madeira_gdf, azores_gdf):
    # Create a Point object from the given coordinates
    point_geom = Point(point[1], point[0])

    # Create a GeoDataFrame for the input point
    point_gdf = gpd.GeoDataFrame(geometry=[point_geom], crs="EPSG:4326")

    # Convert the CRS of the point GeoDataFrame to match the district_parish_gdf CRS
    point_gdf = point_gdf.to_crs(district_parish_gdf.crs)

    # Perform the spatial join with district_parish_gdf
    joined_gdf = gpd.sjoin(point_gdf, district_parish_gdf, how='inner', predicate='within')

    # Check if there is a matching district and parish in district_parish_gdf
    if len(joined_gdf) > 0:
        concelho = joined_gdf['Concelho'].iloc[0]
        distrito = joined_gdf['Distrito'].iloc[0]
        return str(distrito), str(concelho)

    else:
        # Convert the CRS of the point GeoDataFrame to match the madeira_gdf CRS
        point_gdf_madeira = point_gdf.to_crs(madeira_gdf.crs)
        # Perform the spatial join with madeira_gdf
        joined_gdf_madeira = gpd.sjoin(point_gdf_madeira, madeira_gdf, how='inner', predicate='within')
        # Check if there is a matching district and parish in madeira_gdf
        if len(joined_gdf_madeira) > 0:
            concelho = joined_gdf_madeira['Concelho'].iloc[0]
            distrito = joined_gdf_madeira['Ilha'].iloc[0]
            return str(distrito), str(concelho)

        else:
            point_gdf_azores = point_gdf.to_crs(azores_gdf.crs)
            joined_gdf_azores = gpd.sjoin(point_gdf_azores, azores_gdf, how='inner', predicate='within')
            # Check if the point belongs to the Azores region
            if len(joined_gdf_azores) > 0:
                concelho = joined_gdf_azores['Concelho'].iloc[0]
                distrito = joined_gdf_azores['Ilha'].iloc[0]
                return str(distrito), str(concelho)

            # If no matching district and parish found in any region
            print("No matching district for given point:", str(point))
            return '', ''


if __name__ == '__main__':
    print(AZORES_GDF.head())
    print(AZORES_GDF.columns)

    excel_data = process_excel_file(FILE_PATH)
    save_data_to_txt(excel_data, SAVE_PATH)



   # figueira = (40.151, -8.855)


