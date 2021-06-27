import cv2
import sys

def get_frame(video_name):
    video_path = 'Affwild2_Validation_VA_videos/' + 'Validation_Set/' + video_name
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    frame_number = 0

    #if not Path(_OUTPUT_DIR + data + "/frame/").exists():
    #    os.mkdir(_OUTPUT_DIR + data + "/frame/")

    while success:

        video = video_name.split(".")
        data_id = "{}_{}".format(video[0], frame_number)
        # change to my directory that I will create for my frames for all the videos
        # eg a frame will be something like vid131/001.jpg
        output = "AffWild2_Validation_Frames/" + data_id + ".jpg"
        # we take every 20th frame as each one is impractical
        if frame_number % 20 == 0:
            cv2.imwrite(output, image)     # save frame as JPEG file
        success,image = vidcap.read()
        # print('Read a new frame: ', success)
        frame_number += 1

    return frame_number


if __name__ == "__main__":

    subset_videos = ["video94.mp4"]

    # problem there is over a quarter of a million frames here (maybe take every 20th frame)
    videos = ["1-30-1280x720.mp4","8-30-1280x720.mp4","12-24-1920x1080.mp4","13-30-1920x1080.mp4","16-30-1920x1080.mp4",
     "21-24-1920x1080.mp4","25-25-600x480.mp4","26-60-1280x720.mp4","27-60-1280x720.mp4","48-30-720x1280.mp4",
     "53-30-360x480.mp4","72-30-1280x720.mp4","76-30-640x280.mp4","77-30-1280x720.mp4","79-30-960x720.mp4",
     "81-30-576x360.mp4","84-30-1920x1080.mp4","107-30-640x480.mp4","112-30-640x360.mp4","113-60-1280x720.mp4",
     "114-30-1280x720.mp4","115-30-1280x720.mp4","118-30-640x480.mp4","126-30-1080x1920.mp4","127-30-1280x720.mp4",
     "132-30-426x240.mp4","video1.mp4","video54.mp4","video55.mp4","video56.mp4","video57.mp4","video58.mp4",
     "video59.mp4","video60.mp4","video61.mp4","video62.mp4","video63.mp4","video64.mp4","video66.mp4","video67.mp4",
     "video69.mp4","video70.mp4","video71.mp4","video74.mp4","video75.mp4","video76.mp4","video77.mp4","video78.mp4",
     "video79.mp4","video80.mp4","video81.mp4","video82.mp4","video83.mp4","video84.mp4","video85.mp4","video86_1.mp4",
     "video86_2.mp4","video86_3.mp4","video87.mp4","video88.mp4","video89.mp4","video90.mp4","video91.mp4","video92.mp4",
     "video93.mp4","video94.mp4","video95.mp4","video96.mp4"]

    for video in videos:
        get_frame(video)