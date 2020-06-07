import glob
import shutil

for i in glob.glob('*.jpg'):
    dst = "/home/shamim/Downloads/Compressed/face-mask-detector/data_create/images/human_images/dst_1/"
    res = i.split(".")
    if res[0][-1]=="k":
        shutil.move(i, dst)
        print(i)
    