import numpy as np
import cv2

img = cv2.imread('assets/fourCards2.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshold_img = cv2.threshold(gray_img, 190, 255, cv2.THRESH_BINARY)[1]

contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_with_contours = img.copy()

min_contour_area = 1000
target_width, target_height = 200, 300
for i, contour in enumerate(contours):
    contour_area = cv2.contourArea(contour)
    if contour_area > min_contour_area:
        # Draw the contour
        cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 0), 3)

        # Create a mask for the current contour
        mask = np.zeros_like(threshold_img)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

        # Get the corners of the contour
        epsilon = 0.04 * cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, epsilon, True)

        if len(corners) == 4:
            # Sort the corners based on their sum (top-left has the smallest sum, bottom-right has the largest sum)
            corners = np.array(sorted(corners, key=lambda x: np.sum(x, axis=1)))

            # Create source and destination points for top-down view transformation
            src_pts = corners.astype(np.float32)
            dst_pts = np.float32([[0, 0], [target_width, 0], [0, target_height], [target_width, target_height]])

            # Compute the perspective transformation matrix
            perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # Apply the perspective transformation to extract the card
            warped_card = cv2.warpPerspective(img, perspective_matrix, (target_width, target_height))

            # Display the individual card
            cv2.imshow(f'Warped_Card_{i+1}', warped_card)

# Display the image with large contours
cv2.imshow('img_with_large_contours', img_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
