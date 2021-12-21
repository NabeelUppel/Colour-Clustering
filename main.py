import numpy as np
import cv2


def kMeans(K, imName):
    print(f"{K} Colours")
    data = getRGB(imName)
    Centroids = np.zeros((K, 3))
    for i in range(K):
        rgb = np.random.randint(0, 255, 3)
        Centroids[i] = rgb

    eps = 1
    max_it = 15
    oldCentroids = np.full(Centroids.shape, 0)
    eps_arr = np.full(Centroids.shape, eps)
    cur_it = 0
    unused = []
    while cur_it < max_it:
        print("Iteration", cur_it)
        for i in range(len(data)):
            dp = data[i]
            last, dp = dp[-1], dp[:-1]
            col_index = closestCentroid(Centroids, dp)
            dp = np.append(dp, col_index)
            data[i] = dp

        if np.all(Centroids.shape == oldCentroids.shape):
            diff = np.abs(oldCentroids - Centroids)
            print(diff)
            if np.all(diff < eps_arr):
                break
            else:

                oldCentroids = Centroids
                Centroids, unused = newCent(data, K)

        else:
            print(f"Not using all colours in {K} Clusters")
            # Remove unused clusters.
            for u in reversed(unused):
                oldCentroids = np.delete(oldCentroids, u, axis=0)
            while eps_arr.shape[0] > Centroids.shape[0]:
                eps_arr = np.delete(eps_arr, 0, axis=0)

        cur_it += 1

    print("Total Iterations", cur_it)
    return Centroids.astype(int), data


def checkConvergence(new, old, eps):
    x = np.abs(np.subtract(new, old))
    return np.all(x <= eps)


def newCent(data, K):
    Centroids = []

    unused = []
    used = np.unique(data[:, -1])
    for i in range(K):
        if i not in used:
            unused.append(i)

    split = [data[data[:, -1] == k] for k in np.unique(data[:, -1])]
    for arr in split:
        rgbMean = []
        arr = np.delete(arr, -1, axis=1)
        r, c = arr.shape
        for i in range(c):
            x = np.mean(arr[:, i])
            rgbMean.append(x)
        Centroids.append(rgbMean)
    Centroids = np.array(Centroids)

    return Centroids, unused


def closestCentroid(centers, dp):
    distances = []
    for c in centers:
        x = ManhattanDistance(c, dp)
        distances.append(x)
    mn_val = min(distances)
    index = distances.index(mn_val)
    return index


def ManhattanDistance(rgbC, rgbD):
    return np.abs(rgbC - rgbD).sum()


def getRGB(imName):
    image = cv2.imread(imName)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    pixels = height * width
    rgb = image.reshape(pixels, 3)
    neg = np.full(pixels, -1)
    rgb = np.column_stack((rgb, neg))
    return rgb


def reColour(colours, data, K, imName):
    image = cv2.imread(imName)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    org_shape = image.shape
    height, width = org_shape[0], org_shape[1]
    pixels = height * width
    image = image.reshape(pixels, 3)

    for i in range(len(image)):
        index, rgb = data[i][-1], data[i][:-1]
        if np.array_equal(image[i], rgb):
            image[i] = colours[index]

    image = image.reshape(org_shape)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    file = open("Tierney/Tierney.txt", "a")
    shape = colours.shape
    file.write(f"{K}_Clusters")
    file.write("\n")
    file.write("No. of Colours: " + str(shape))
    file.write("\n")
    file.write("Colours: " + str(colours))
    file.write("\n")
    file.close()

    name = f"{K}_Colours"
    cv2.imwrite(name + ".jpg", image)


def main():
    image1 = "Peppers/peppers.bmp"
    image2 = "Tierney.jpg"
    for i in range(4, 5):
        K = 2 ** i
        Colours, data = kMeans(K, image2)
        reColour(Colours, data, K, image2)


if __name__ == '__main__':
    main()
