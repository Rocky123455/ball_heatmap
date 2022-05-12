import io
import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from moviepy.video.io.bindings import mplfig_to_npimage

def process(img):
    #plt.imshow(img), plt.show()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 10)
    #img_canny = cv2.Canny(img_blur, 5, 40)
    img_gray[img_gray < 20] = 0
    kernel = np.ones((5, 5))
    #img_blur = cv2.erode(img_blur, kernel, iterations=1)

    return img_gray

def get_contours(img, img_original):
    img_contours = img_original.copy()
    contours, hierarchies = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    balls = []
    for c in contours:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        balls.append((cX, cY))
        cv2.circle(img_contours, (cX, cY), 10, (255, 255, 255), -1)
    #ggplt.imshow(img_contours), plt.show()
    #cv2.drawContours(img_contours, contours, -1, (0, 255, 0), -1)
    # If you want to omit smaller contours, loop through the detected contours, and only draw them on the image if they are at least a specific area. Don't forget to remove the line above if you choose the below block of code.
            # for cnt in contours:
            #     if cv2.contourArea(cnt) > 400:
    #        cv2.drawContours(img_contours, [cnt], -1, (0, 255, 0), -1)

    return balls

cap = cv2.VideoCapture("vid.mp4")

cap.set(cv2.CAP_PROP_POS_FRAMES, 1)




i = 0

success,img1 = cap.read()
#img6 = img1.copy()
#img5 = img1.copy()
#img4 = img1.copy()
#img3 = img1.copy()
img2 = img1.copy()


output_video_path = "video_output.avi"
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (1152, 864))




list_diffs = []
points_history = []
while success :

    diff = process(cv2.absdiff(img1, img2))

    kernel = np.ones((5, 5))




    sum_diff = diff
    for d in list_diffs:
        sum_diff = sum_diff + d

    #sum_diff = cv2.dilate(sum_diff, kernel, iterations=2)
    #sum_diff = cv2.erode(sum_diff, kernel, iterations=2)
    sum_diff[sum_diff > 0] = 255


    def is_contour_bad(c):
        # approximate the contour

        peri = cv2.arcLength(c, True)

        if peri == 0:
            return False

        approx = cv2.contourArea(c) / peri
        # the contour is 'bad' if it is not a rectangle
        return not approx < 6




    copy_diff =sum_diff.copy()

    copy_diff = cv2.dilate(copy_diff, kernel, iterations=1)
    #copy_diff = cv2.erode(copy_diff, kernel, iterations=1)
    cnts = cv2.findContours(copy_diff, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    mask = np.ones(sum_diff.shape[:2], dtype="uint8") * 255
    # loop over the contours
    for c in cnts:
        # if the contour is bad, draw it on the mask
        if is_contour_bad(c):
            cv2.drawContours(mask, [c], -1, 0, -1)

    sum_diff = cv2.bitwise_and(sum_diff, sum_diff, mask=mask)
    #diff = cv2.bitwise_and(diff, diff, mask=mask)
    #if len(list_diffs) > 10:
        #plt.imshow(sum_diff), plt.show()


    lines = cv2.HoughLinesP(
        sum_diff,  # Input edge image
        cv2.HOUGH_PROBABILISTIC,  # Distance resolution in pixels
        np.pi / 720,  # Angle resolution in radians
        threshold=40,  # Min number of votes for valid line
        minLineLength=20,  # Min allowed length of line
        maxLineGap=50  # Max allowed gap between line for joining them
    )

    #if not lines is None:
   #    for points in lines:
   #         x1, y1, x2, y2 = points[0]

            #cv2.line(opencvImage, (x1, y1), (x2, y2), (0, 255, 0), 2)




    #plt.imshow(opencvImage, plt.show())
    diff2 = cv2.bitwise_and(diff, diff, mask=mask)
    diff2[diff2 > 0] = 255

    circles = cv2.HoughCircles(diff, cv2.HOUGH_GRADIENT, dp=4, minDist=40, param1=50, param2=2, minRadius=6,
                               maxRadius=12)
    opencvImage = cv2.cvtColor(diff, cv2.COLOR_RGB2BGR)
    if circles is not None:
        for c in circles[0]:
            print(c)
            #cv2.circle(opencvImage, (int(c[0]), int(c[1])), 3, (0, 255, 0), 2)
        points = [ (int(c[0]), int(c[1])) for c in circles[0]]
        points_history.append(points)


        print()
    else:
        points_history.append([])

    total = []
    for p in points_history[-30:]:
        total = total + p

    x = [a[0] for a in total]
    y = [a[1] for a in total]
    area = [3 for i in range(len(x) + 2)]  # 0 to 15 point radii
    x.append(0)
    y.append(0)
    x.append(output_width)
    y.append(output_height)



    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x, y, s=area, alpha=0.5)

    def get_img_from_fig(fig, dpi=180):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


    # you can get a high-resolution image as numpy array!!
    plot_img_np = get_img_from_fig(fig)



    opencvImage = plot_img_np
    opencvImage = cv2.cvtColor(opencvImage, cv2.COLOR_RGB2BGR)


    #plt.imshow(opencvImage)
    #plt.show()

    output_video.write(opencvImage)

    #combinations = get_contours(diff, img1)


    #img6 = img5.copy()
    #img5 = img4.copy()
    #img4 = img3.copy()
    #img3 = img2.copy()
    img2 = img1.copy()

    list_diffs.append(diff)
    if len(list_diffs) > 20:
        list_diffs= list_diffs[1:]

    success, img1 = cap.read()
    i+=1

    print(i)


cap.release()
output_video.release()
