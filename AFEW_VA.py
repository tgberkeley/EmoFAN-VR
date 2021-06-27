import json
from pathlib import Path
import os

video_number = "017"

output_path = video_number + "/annotations/" + ".txt"

if not Path(video_number + "/annotations").exists():
    os.mkdir(video_number + "/annotations")

with open(video_number + "/" + video_number +".json", "r") as read_file:
    data = json.load(read_file)

    frames = data['frames']
    print(frames.keys())
    for key, value in frames.items():
        print(key)
        print(value.keys())

        with open(output_path, 'a') as f:
            string = (f"{key}, {value['arousal']}, {value['valence']}\n")
            f.write(string)
            f.close()

