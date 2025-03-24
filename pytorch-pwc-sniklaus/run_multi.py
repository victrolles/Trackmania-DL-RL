import subprocess
import os

interval = []       # [start, end] if empty, all files in image directory will be used
image_directory = "images_firstP"
image_name = "ezgif-frame-"

if __name__ == "__main__":
    input_file = []

    if not interval:
        input_file = os.listdir(image_directory)
    else :
        for i in range(interval[0], interval[1]+1):
            if i < 10:
                input_file.append(f"{image_directory}/{image_name}00{i}.png")
            elif i < 100:
                input_file.append(f"{image_directory}/{image_name}0{i}.png")
            else:
                input_file.append(f"{image_directory}/{image_name}{i}.png")

    if len(input_file) < 2:
        print("Not enough images to create flows")
        exit()

    for i in range(len(input_file) - 1):
        # generate flow between two images
        print(f"Creating flow between {image_directory}/{input_file[i]} and {image_directory}/{input_file[i+1]}")
        result_flo = subprocess.run(
            ["python", "generate_out_flow.py", "--model", "default", "--one", f"{image_directory}/{input_file[i]}", "--two", f"{image_directory}/{input_file[i+1]}"], capture_output=True, text=True)
        if result_flo.stderr:
            print(f"Error while creating flow between {input_file[i]} and {input_file[i+1]}")
            print(result_flo.stderr)


        # convert flow to image
        result_ftoi = subprocess.run(["python", "flow_to_image.py"], capture_output=True, text=True)
        if result_ftoi.stderr:
            print(f"Error while converting flow to image")
            print(result_ftoi.stderr)




