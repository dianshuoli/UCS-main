import os
import json

def get_image_paths(folder_path):
    image_paths = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp')):
                relative_path = "./data_demo/images/" + os.path.relpath(os.path.join(root, file), folder_path)
                image_paths[relative_path] = relative_path
    return image_paths

folder_path = "images"
image_paths_dict = get_image_paths(folder_path)

with open('label2image_test.json', 'w') as json_file:
    json_file.write("{\n")
    for index, (key, value) in enumerate(image_paths_dict.items()):
        json_file.write(f'"{key}": "{value}"')
        if index < len(image_paths_dict) - 1:
            json_file.write(",\n")
        else:
            json_file.write("\n")
    json_file.write("}")