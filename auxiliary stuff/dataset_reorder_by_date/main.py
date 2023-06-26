import datetime

FILE_PATH = "C:\\Users\\Hugo\\Desktop\\main_project\\october_fires.txt"


def read_data_file(file_path):
    with open(file_path, 'r') as rf:
        lines = rf.readlines()[1:]  # skip the first line

    # list of tuples & arrays
    data = []

    # go over each line and partition its data
    for line in lines:
        line = line.strip().split(';')
        submission_id = int(line[0])
        submission_date = datetime.datetime.strptime(line[1], '%d/%m/%Y %H:%M')
        user_id = int(line[2])
        user_rating = int(line[3])  # rating is both the quality and quantity of the submissions
        fire_verified = int(line[4])
        smoke_verified = int(line[5])
        lat = float(line[6])
        lon = float(line[7])
        text_district = line[8]
        text_parish = line[9]
        text_keywords = line[10]

        # join current iteration line into the input array
        data.append((submission_id, submission_date, user_id, user_rating, fire_verified, smoke_verified, lat, lon,
                     text_district, text_parish, text_keywords))

    return data


def reorder_data_by_submission_date(data):
    sorted_data = sorted(data, key=lambda x: x[1])  # Sort based on submission_date (index 1)
    return sorted_data



def save_data_to_txt(data, file_path):
    with open(file_path, 'w') as file:
        # Write the header
        header = "sub_id;date/time;user_id;user_rating;fire_verified;smoke_verified;lat;lon;text_district;text_loc;text_keywords\n"
        file.write(header)

        # Write the data rows
        for row in data:
            submission_id, submission_date, user_id, user_rating, fire_verified, smoke_verified, lat, lon, text_district, text_parish, text_keywords = row

            submission_date_str = submission_date.strftime('%d/%m/%Y %H:%M')

            row_str = f"{submission_id};{submission_date_str};{user_id};{user_rating};{fire_verified};{smoke_verified};{lat};{lon};{text_district};{text_parish};{text_keywords}\n"

            file.write(row_str)

    print(f"Data saved to file: {file_path}")


if __name__ == '__main__':
    data = read_data_file(FILE_PATH)

    ordered_data = reorder_data_by_submission_date(data)

    save_data_to_txt(ordered_data, "C:\\Users\\Hugo\\Desktop\\dataset_reorder_by_date\\october_fires.txt")

