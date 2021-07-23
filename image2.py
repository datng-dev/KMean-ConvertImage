# cach khac
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
import numpy

img = plt.imread("tree.jpg") #doc img (chuyen than tuple)

height = img.shape[0] #Lay ra height
width = img.shape[1]

img = img.reshape(height*width, 3) #chuyen image thanh mot array chua cac array la cac pixel (duoi img thanh hang doc)

#print(img.shape)

#print(img)

kmeans = KMeans(n_clusters=4).fit(img)

labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_

#print(labels)
#print(clusters)

img2 = numpy.zeros((height, width, 3), dtype=numpy.uint8) # tao ra mot image hoan toan chua duỗi

index = 0
for i in range(height):
    for j in range(width):
        label_of_pixel = labels[index]
        img2[i][j] = clusters[label_of_pixel]
        index += 1

# Tat frame
fig = plt.figure(frameon=False)

# Cho buc anh lap day khung hinh
ax = plt.Axes(fig, [0., 0., 1., 1.])

# Tat truc toa do
ax.set_axis_off()
fig.add_axes(ax)

# Hien thi anh
plt.imshow(img2)
plt.show()