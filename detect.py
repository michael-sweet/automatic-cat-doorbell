from transformers import YolosImageProcessor, YolosForObjectDetection
from picamera2 import Picamera2
from PIL import Image
import configparser
import requests
import datetime
import shutil
import torch
import time
import os

os.environ["LIBCAMERA_LOG_LEVELS"] = "3"

config = configparser.ConfigParser()
config.read('config.ini')
image_folder = config.get('Settings', 'image_folder')
notification_url = config.get('Settings', 'notification_url')
yolos_model = config.get('Settings', 'yolos_model')
min_notification_interval_minutes = int(config.get('Settings', 'min_notification_interval_minutes'))

last_notification_time = 0
model = YolosForObjectDetection.from_pretrained(yolos_model)
image_processor = YolosImageProcessor.from_pretrained(yolos_model)

cam = Picamera2()
filename = image_folder + "frame.jpg"

while True:
    # Take a photo and save to file
    cam.start_and_capture_file(filename, show_preview=False)

    # Ensure we do not spam notifications
    if time.time() - last_notification_time > min_notification_interval_minutes * 60:
        # Open saved image
        image = Image.open(filename)
        # Resize image for model
        resized_img = image.resize((640, 480))
        resized_img.save(image_folder + "frame_resized.jpg")

        # Set model parameters
        inputs = image_processor(images=resized_img, return_tensors="pt")
        target_sizes = torch.tensor([resized_img.size[::-1]])
        outputs = model(**inputs)

        # Attempt to detect objects
        results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
        for label in results["labels"]:
            # Cat spotted!
            if model.config.id2label[label.item()] == "cat":
                # Copy image to convenient places
                shutil.copyfile(filename, image_folder + "last_cat_sighting.jpg")
                shutil.copyfile(filename, image_folder + "cat_" +  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  + ".jpg")
                # Send notification
                requests.get(notification_url)
                last_notification_time = time.time()
                print("Cat spotted!")

    # Wait 10 seconds before checking again
    time.sleep(10)