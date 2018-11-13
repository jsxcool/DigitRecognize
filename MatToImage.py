import numpy as np
import csv
import matplotlib.pyplot as plt

### Matrix to Image
X = []  # X is a list of pixel matrix
i = 0
with open('/home/jiang/Jupyter/allDigit/test.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		if reader.line_num == 1:
			continue;
		# row is a pixel vector(dtype=list)
		X.append(np.array(row, dtype=int).reshape(28,28))
		i += 1
		if i > 8:
			break;

fig, axes = plt.subplots(3, 3)
for i in range(3):
    index = i;
    for j in range(3):
        axes[i][j].imshow(X[3*index+j], cmap=plt.get_cmap('gray'), vmin=0, vmax=255) 


### Image to Matrix
from PIL import Image

fname = '/home/jiang/Jupyter/7.jpg'
image = Image.open(fname).convert("L")
arr = np.asarray(image)
#plt.imshow(arr, cmap='gray')
plt.show()
