import os 
import csv
from pathlib import Path

if __name__ == '__main__':
  with open('./anontations/faces.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
      try:

        user_dir = os.path.join("./datasets", row["id_user"])
        # create folder
        os.makedirs(user_dir, exist_ok=True)

        # move file and group by id user
        file_path = Path(row["image_path"])
        print("move", os.path.join("./datasets", "align", file_path.name), os.path.join( user_dir, file_path.name))
        os.rename( os.path.join("./datasets", "align", file_path.name), os.path.join( user_dir, file_path.name), )
      
      except Exception as e:
        print(f"error ${e}")
      