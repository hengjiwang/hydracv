import os
import tiffstack2avi

def traverse(path, suffix):
    image_formats = [suffix]
    for (root, dirs, files) in os.walk(path):  
        for filename in files:
            if os.path.splitext(filename)[1] in image_formats:
                print(os.path.join(root,filename))


if __name__ == "__main__":
    traverse("/media/hengji/DATA/Data/Documents/szymanski_data", ".tif")