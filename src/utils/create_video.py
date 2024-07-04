import os
import glob
import numpy as np
from PIL import Image
from moviepy.editor import ImageSequenceClip

def create_image(img_path1: str, img_path2: str, output_filename: str) -> str:
    # Open the two input images
    image1 = Image.open(img_path1)
    image2 = Image.open(img_path2)

    # Get the dimensions of the input images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Calculate the width and height of the output image
    output_width = width1 + width2
    output_height = max(height1, height2)

    # Create a new blank image with the calculated dimensions
    output_image = Image.new('RGB', (output_width, output_height))

    # Paste the first image on the left side of the output image
    output_image.paste(image1, (0, 0))

    # Paste the second image on the right side of the output image
    output_image.paste(image2, (width1, 0))

    # Save the combined image
    output_image.save(output_filename)


def create_mp4(path: str, video_name: str, duration=100):
    """
    :param path: path where experiment data was saved
    :param video_name: output video name
    :param duration: duration of each frame in ms
    :return:
    """
    # Provide the folder path containing the images
    folder_path = os.path.join(path, 'figures')
    # Provide the output path for the video
    output_video = os.path.join(path, video_name)
    output_video_mode = os.path.join(path, "map_video.mp4")
    # Temp folder for images joined
    temp_path = os.path.join(path, "temp")

    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                   file.endswith(".png") or file.endswith(".jpg")]
    image_paths.sort()
    image_paths = np.asarray(image_paths, dtype=str)
    mask_mean = np.zeros(image_paths.shape[0], dtype=bool)
    mask_mode = np.zeros(image_paths.shape[0], dtype=bool)
    mask_filters = np.zeros(image_paths.shape[0], dtype=bool)

    # Create temp folder for images
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    mask_mean = [True if 'main' in path else mask_mean[i] for i, path in enumerate(image_paths)]
    mask_mode = [True if 'map' in path else mask_mode[i] for i, path in enumerate(image_paths)]
    mask_filters = [True if 'filters' in path else mask_filters[i] for i, path in enumerate(image_paths)]
    
    mean_paths = image_paths[mask_mean]
    mode_paths = image_paths[mask_mode]
    filter_paths = image_paths[mask_filters]

    output_images = []
    output_images_mode = []

    it = 0
    for m_path, mode_path, f_path in zip(mean_paths, mode_paths, filter_paths):
        output_filename = f"{temp_path}/combined_image_{it}.png"
        output_filename_mode = f"{temp_path}/combined_image_mode_{it}.png"
        create_image(m_path, f_path, output_filename)
        create_image(mode_path, f_path, output_filename_mode)
        output_images.append(output_filename)
        output_images_mode.append(output_filename_mode)
        it += 1

    # Create an MP4 video using the list of output image filenames
    clip = ImageSequenceClip(output_images, durations=[duration / 1000] * len(output_images))
    clip.write_videofile(output_video, fps=1 / duration * 1000, codec="libx264", audio=False)
    # Create Mode video
    clip = ImageSequenceClip(output_images_mode, durations=[duration / 1000] * len(output_images_mode))
    clip.write_videofile(os.path.join(path, output_video_mode), fps=1 / duration * 1000, codec="libx264", audio=False)

    # Remove the temporary image files (optional)
    for file in glob.glob(f'{temp_path}/*.png'):
        if file not in {output_video, output_video_mode}:
            os.remove(file)
