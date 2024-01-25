import numpy as np
import cv2
import os


def get_card_info(card_image):
    crop_img = card_image[0:90, 0:30]
    crop_img = cv2.resize(crop_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    crop_img = cv2.threshold(crop_img, 190, 255, cv2.THRESH_BINARY_INV)[1]

    number_img = crop_img[0:100, 0:60]
    suit_img = crop_img[100:170, 0:60]

    contours_number, _ = cv2.findContours(number_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_number:
        contours_suit, _ = cv2.findContours(suit_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_suit:
            x_number, y_number, w_number, h_number = cv2.boundingRect(contours_number[0])
            x_suit, y_suit, w_suit, h_suit = cv2.boundingRect(contours_suit[0])

            region_number = number_img[y_number:y_number + h_number, x_number:x_number + w_number]
            region_suit = suit_img[y_suit:y_suit + h_suit, x_suit:x_suit + w_suit]

            resized_region_number = cv2.resize(region_number, (70, 110), interpolation=cv2.INTER_LINEAR)
            resized_region_suit = cv2.resize(region_suit, (70, 110), interpolation=cv2.INTER_LINEAR)

            card_num = find_best_match(resized_region_number, 'num_img', min_similarity=0.8)
            card_suit = find_best_match(resized_region_suit, 'suit_img', min_similarity=0.8)

            if card_num is not None and card_suit is not None:
                card_id = card_num + card_suit
                return card_id, card_num

    return None, None


def find_best_match(query_img, folder, min_similarity=0.8):
    best_match = None
    best_score = float('-inf')

    for template_file in os.listdir(folder):
        template_path = os.path.join(folder, template_file)
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        result = cv2.matchTemplate(query_img, template_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score and max_val >= min_similarity:
            best_score = max_val
            best_match = os.path.splitext(template_file)[0]
    return best_match


def check_seen(name):
    if name not in id_set:
        id_set.add(name)
        return True
    else:
        return False


def get_card_owner(corners):
    if len(corners) > 0 and corners[0][0, 1] < video_height // 3:
        return "dealer"
    else:
        return "player"


def count_update(owner, value):
    global player_count
    global dealer_count

    if owner == "dealer":
        dealer_count += card_values.get(value)
    else:
        player_count += card_values.get(value)


def process_video_frame(frame):
    global running_count
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray_img, 190, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = frame.copy()

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > 1000:
            cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 0), 3)
            mask = np.zeros_like(threshold_img)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            epsilon = 0.04 * cv2.arcLength(contour, True)
            corners = cv2.approxPolyDP(contour, epsilon, True)

            if len(corners) == 4:
                corners = np.array(sorted(corners, key=lambda x: np.sum(x, axis=1)))
                src_pts = corners.astype(np.float32)
                dst_pts = np.float32([[0, 0], [target_width, 0], [0, target_height], [target_width, target_height]])
                perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped_card = cv2.warpPerspective(frame, perspective_matrix, (target_width, target_height))
                card_index, card_number = get_card_info(warped_card)
                if card_index is not None and check_seen(card_index):
                    running_count += num_mapping.get(card_number)
                    card_owner = get_card_owner(corners)
                    count_update(card_owner, card_number)

    cv2.putText(img_with_contours, f"Running Count: {running_count}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    count_text = f"Running Count: {running_count}   Player Count: {player_count}   Dealer Count: {dealer_count}"
    cv2.putText(img_with_contours, count_text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Contours', img_with_contours)


video_path = 'assets/card_dealing.mp4'
cap = cv2.VideoCapture(video_path)
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
target_width, target_height = 200, 300
running_count = 0
dealer_count = 0
player_count = 0
id_set = set()
num_mapping = {
    "two": 1, "three": 1, "four": 1, "five": 1, "six": 1,
    "seven": 0, "eight": 0, "nine": 0,
    "ten": -1, "jack": -1, "queen": -1, "king": -1, "ace": -1
}

card_values = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "jack": 10, "queen": 10, "king": 10, "ace": 10
}

while cap.isOpened():
    ret, vid_frame = cap.read()
    if not ret:
        break

    process_video_frame(vid_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Updated ID Set:", id_set)
print("Running Count:", running_count)
