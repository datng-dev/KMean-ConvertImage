import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
import numpy

img = plt.imread("khoai.jpg") #doc img (chuyen than tuple)


height = img.shape[0] #Lay ra height
width = img.shape[1]

img = img.reshape(height*width, 3) #chuyen image thanh mot array chua cac array la cac pixel (duoi img thanh hang doc)

# print(img.shape)

# print(img)

kmeans = KMeans(n_clusters=4).fit(img)

labels = kmeans.predict(img)
print(labels)
clusters = kmeans.cluster_centers_
print(clusters)

#print(labels)
#print(clusters)

img2 = numpy.zeros_like(img) # tao bien moi toan la so 0 co chieu giong img nam trong mot cai array (ko phai la tuple)
print(img2.shape)

for i in range(len(img2)):
    img2[i] = clusters[labels[i]]

img2 = img2.reshape(height, width, 3) # Khoi phuc buc anh tu chieu doc

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