import pandas
from PIL import Image
from PIL import ImageEnhance
import os

df = pandas.read_csv(r'./A_Z Handwritten Data.csv')
parent_dir = './dataset'
mode = 0o666

for k in range(len(df)):
    row_list = df.iloc[k].tolist()
    alphabet = row_list[0]
    cur_dir = os.path.join(parent_dir,str(alphabet))
    if not os.path.exists(cur_dir):
        os.mkdir(cur_dir,mode)
    l = row_list[1:len(row_list)]
    for i in range(len(l)):
        l[i] = abs(255 - l[i])
    image = Image.new('L', (28, 28))
    image.putdata(l)
    cur_sharp = ImageEnhance.Sharpness(image)
    new_sharp = 10
    img = cur_sharp.enhance(new_sharp)
    os.chdir(cur_dir)
    img.save(str(k)+".jpg")
