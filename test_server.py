import requests
import time
import argparse
import os
import sys

API_URL = "https://api.cortex.cerebrium.ai/v4/p-a793d1c5/imageclassifier/predict"
API_KEY  = "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWE3OTNkMWM1IiwiaWF0IjoxNzQ4Njk4MDE1LCJleHAiOjIwNjQyNzQwMTV9.NwKFjUQExuwOFIIiXflsIuDE8V003-xqrzi3nD29Tb8NJPeIIMRVO3_Tjt5tbjwZdP02JJINBHZbwumYmgnv35WI2Yh_xBLnpTysPa3StCumHeXOMWPWkjSxXHJWGj5cQepKDHWcivZipR8inmTwj96AiRN9kJmGuwVZ7A04qfhEh-ksy3cCRdc3vbQj4m99U6vWBLsz39D2e_aVn6CYkNwI6YPZwEP8dR_PoSzUkCJvEY28qoLtPrd_9orHECyngRnu_sE3GYThSa31VlZjTmdD7IgaqhwPRsP-JGZ9LK7Z2WwZmZcRnB4WAn1e5wmAkKPyajiEgZKh7yp7Xfsx2g"


def main():
    parser = argparse.ArgumentParser(description="Cerebrium Image Classifier")
    parser.add_argument("image_path", help="Path to the image file")
    args = parser.parse_args()

    image_path = args.image_path

    if not os.path.isfile(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        sys.exit(1)

    with open(image_path, "rb") as f:
        files = {"model_bytes_str": ("image.jpg", f, "image/jpeg")}
        headers = {"Authorization": API_KEY}

        start_time = time.time()
        response = requests.post(API_URL, files=files, headers=headers)
        end_time = time.time()

    print(response.status_code)
    print(response.json())
    print("Elapsed (client timing):", end_time - start_time, "seconds")
    print("Elapsed (server reported):", response.elapsed.total_seconds(), "seconds")

if __name__ == "__main__":
    main()

