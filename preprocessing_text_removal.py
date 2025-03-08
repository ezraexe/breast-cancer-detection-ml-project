import cv2
import os
import numpy as np
from keras_ocr import pipeline

# need tensorflow v 2.15 not most recent keras does not like 2.18

def inpaint_text(img_path, pipeline):

    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Could not read image '{img_path}'.")
        return None

    # keras-OCR expects RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # detect text 
    prediction_groups = pipeline.recognize([img])

    # mask to cover text 
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        # bounding box coordinates
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        # points that are gonna be covered
        points = np.array([[int(x0), int(y0)], [int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)]], dtype=np.int32)

        # fill in mask
        cv2.fillPoly(mask, [points], 255)

    # mask is black
    mask = 255 - mask

    # put mask on image
    inpainted_img = cv2.bitwise_and(img, img, mask=mask)

    return inpainted_img

def process_folder(input_folder, output_folder):

    # create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for subdir, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                #  image path
                img_path = os.path.join(subdir, filename)

                # process the image
                inpainted_img = inpaint_text(img_path, pipeline)

                # check if was successful
                if inpainted_img is not None:
                    output_filename = os.path.join(output_folder, filename)

                    # save the image
                    cv2.imwrite(output_filename, inpainted_img)
                    print(f"Processed image: {img_path}")
                else:
                    print(f"Error processing image: {img_path}")

if __name__ == "__main__":
    # load Keras-OCR pipeline
    pipeline = pipeline.Pipeline()

    # paths - change to actual paths when implementing :)
    input_folder = r"C:\Users\you\file"
    output_folder = r"C:\Users\you\again"
    process_folder(input_folder, output_folder)