import datetime
import random

import pandas as pd
import unicodedata

FILE_PATH = "C:\\Users\\Hugo\\Desktop\\read_convert_csv\\Tweets_incendio_completed.csv"
PARISHES_PATH = "C:\\Users\\Hugo\\Desktop\\read_convert_csv\\Parishes.txt"
SAVE_PATH = "C:\\Users\\Hugo\\Desktop\\read_convert_csv\\october_fires.txt"

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


def process_text(text):
    # Split the text into words
    words = text.split()

    # Remove words with less than 2 characters
    processed_words = [word for word in words if len(word) >= 2]

    # Check if words are in the lists of districts, parishes, and hazards
    district_words = [word for word in processed_words if word in DISTRICTS]
    text_district = ' '.join(list(set(district_words)))  # Remove duplicates

    # Exclude district words from parish words
    remaining_words = [word for word in processed_words if word not in DISTRICTS]
    parish_words = [word for word in remaining_words if word in PARISHES]
    text_parish = ' '.join(list(set(parish_words)))  # Remove duplicates

    text_hazards = list(set([word for word in remaining_words if word in HAZARDS]))  # Remove duplicates

    # Concatenate multiple hazards with a "-"
    text_hazard = '-'.join(text_hazards)

    # Add a random hazard with a 10% chance
    if random.random() < 0.2 and HAZARDS:
        remaining_hazards = [hazard for hazard in HAZARDS if hazard not in text_hazards]
        if remaining_hazards:
            random_hazard = random.choice(remaining_hazards)
            if text_hazard:
                text_hazard += f'-{random_hazard}'
            else:
                text_hazard = random_hazard

    return text_district, text_parish, text_hazard

def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df


def read_excel_file(file_path):
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
            user_rating = random.randint(0, 20)  # Assign a random rating between 0 and 20
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

        # Process the text column
        text = row['Text']
        text_district, text_parish, text_keywords = process_text(text)

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


if __name__ == '__main__':
    excel_data = read_excel_file(FILE_PATH)

    # Print the first 10 lines
    for line in excel_data[:20]:
        print(line)

    save_data_to_txt(excel_data, SAVE_PATH)
