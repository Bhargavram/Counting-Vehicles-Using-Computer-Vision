import skvideo.io
import cv2
import random
import numpy as np
import skvideo
import os
import logging
import logging.handlers
import random
import skvideo
import utils
cv2.ocl.setUseOpenCL(False)
random.seed(123)


from pipeline import (
    ProcessPipelineRunner,
    ContourDetection,
    Vis,
    write_csv,
    VehicleCounter)


SHAPE = (720, 1280)
EXIT_PTS = np.array([
    [[732, 720], [732, 590], [1280, 500], [1280, 720]],
    [[0, 400], [645, 400], [645, 0], [0, 0]]
])

def train_bg_subtractor(instance, caps, num=500):
    i = 0
    for frame in caps:
        instance.apply(frame, None, 0.001)
        i += 1
        if i >= num:
            return caps


Image_dir = "./out"
Vid_src = "input.mp4"

def main():
    skvideo.setFFmpegPath("D:\\Computer Vision\\Homework1\\opencv_traffic_counting\\ffmpeg-20190519-fbdb3aa-win64-static\\bin")
    log = logging.getLogger("main")

    base = np.zeros(SHAPE + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]


    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, detectShadows=True)

    pipeline = ProcessPipelineRunner(pipeline=[
        ContourDetection(bg_subtractor=bg_subtractor,
                         save_image=True, image_dir=Image_dir),

        VehicleCounter(exit_masks=[exit_mask], y_weight=2.0),
        Vis(image_dir=Image_dir),
        write_csv(path='./', name='report.csv')
    ], log_level=logging.DEBUG)


    cap = skvideo.io.vreader(Vid_src)

    # skip num frames and train the background subtractor it will identify the background for all the frames
    bg_Subtractor_train(bg_subtractor, cap, num=1000)

    _frame_number = -1
    frame_number = -1
    for frame in cap:
        if not frame.any():
            log.error("Frame capture failed, stopping...")
            break


        _frame_number += 1

        if _frame_number % 2 != 0:
            continue


        frame_number += 1


        pipeline.set_context({
            'frame': frame,
            'frame_number': frame_number,
        })
        pipeline.run()
def bg_Subtractor_train(inst, cap, num=500):
    i = 0
    for frame in cap:
        inst.apply(frame, None, 0.001)
        i += 1
        if i >= num:
            return cap

if __name__ == "__main__":
    log = utils.init_logging()

    if not os.path.exists(Image_dir):
        log.debug("Creating image directory `%s`...", Image_dir)
        os.makedirs(Image_dir)

    main()
