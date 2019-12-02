# from raw jpg to 300*300 jpg

from PIL import Image
import os

dirname = './landscape/'
proc_output_dir = './proc_out128'
cnt = 0
for filename in os.listdir(dirname):
    try:
        filePath = os.path.join(dirname, filename)
        with Image.open(filePath) as img:
            img = img.resize((128, 128))
            outFile = os.path.join(proc_output_dir, str(cnt))+'.jpg'
            img = img.save(outFile)
            cnt += 1
    except:
        pass


