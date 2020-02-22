import os
from PIL import Image

data_path = "./1st_followup/"

output_path = ".//"

file_name = os.listdir(data_path)

filelist = [os.path.join(data_path, file) for file in file_name]

i = 0

for file in filelist:
    img = Image.open(file)
    img.thumbnail((256,256),Image.ANTIALIAS)
    img.save(file)