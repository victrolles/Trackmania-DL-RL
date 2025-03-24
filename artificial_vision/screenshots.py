import mss
import time
import os
import d3dshot

OUTPUT_DIR = "output_screenshots_"
TESTED = "d3d"

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    output_dir = OUTPUT_DIR + TESTED + "/"
    create_dir(output_dir)

    start = time.time()
    frames = 0

    if TESTED == "mss":
        with mss.mss() as sct:
            for i in range(30):
                output = os.path.join(output_dir, f"capture_{i}.png")

                screen = sct.monitors[2]
                screenshot = sct.grab(screen)
                #mss.tools.to_png(screenshot.rgb, screenshot.size, output=output)
                frames += 1

    elif TESTED == "d3d":
        d = d3dshot.create(capture_output="pil")

        for i in range(100):
            output = os.path.join(output_dir, f"capture_{i}.png")

            img = d.screenshot()
            #img.save(output)

            frames += 1
            print(i)


    print(f"FPS pour {TESTED} :", frames / (time.time() - start))