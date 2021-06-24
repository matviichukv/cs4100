import csv 
import os

healthy_imgs = []
unhealthy_imgs = []

with open("plant_dataset/train.csv", newline='') as csv_file:
  with open("plant_dataset/train_clean.csv", 'w') as clean_csv_file:
    reader = csv.reader(csv_file, delimiter = ',')
    # ignore the first row
    next(reader)
    # write the headers
    clean_csv_file.write("image_id, healthy\n")

    for row in reader:
      if row[1] == '1':
        clean_csv_file.write(f"{row[0]},1\n")
        os.system(f"cp plant_dataset/images/{row[0]}.jpg plant_dataset/images/healthy")
        healthy_imgs.append(row[0])
      else:
        clean_csv_file.write(f"{row[0]},0\n")
        os.system(f"cp plant_dataset/images/{row[0]}.jpg plant_dataset/images/unhealthy")
        unhealthy_imgs.append(row[0])
      os.system(f"rm plant_dataset/images/{row[0]}.jpg")
      

  print(f"healthy {len(healthy_imgs)}, unhealthy: {len(unhealthy_imgs)}")