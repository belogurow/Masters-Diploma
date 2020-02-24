import cv2

if __name__ == "__main__":
    # img = cv2.imread('resources/merlischachen_result/stitching.jpg.png')
    gray_img = cv2.imread('resources/merlischachen_result/stitching.jpg.png', 0)
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 100, 200)

    cv2.imwrite("edges1.jpg", edges)

    ret, thresh = cv2.threshold(gray_img, 3, 255, cv2.THRESH_BINARY)
    cv2.imwrite("edges2.jpg", thresh)

    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv_color1 = np.asarray([240, 100, 100])  # white!
    # hsv_color2 = np.asarray([30, 255, 255])  # yellow! note the order
    #
    # mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)

    # plt.imshow(mask, cmap='gray')  # this colormap will display in black / white
    # plt.show()

    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #
    # plt.show()
