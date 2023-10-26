import numpy as np
import cv2

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


person_image = cv2.imread(r"C:\\Users\\felic\Desktop\\virtual-try-on-app\\cloth-preprocessing\\049433_0.jpg")
mask = cv2.imread(r"C:\\Users\\felic\Desktop\\virtual-try-on-app\\cloth-preprocessing\\049433_0.png", cv2.IMREAD_GRAYSCALE)

background = np.full(person_image.shape, (144,158,250), dtype='uint8')

transparent = np.zeros((person_image.shape[0], person_image.shape[1], 4), dtype=np.uint8)
transparent[:,:,0:3] = person_image
transparent[:, :, 3] = mask

no_bg_person = transparent

merged_image = overlay_transparent(background, no_bg_person, 0, 0)

cv2.imwrite('imgNoBG_0.jpg', merged_image)
