import os


def number_files(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    for i, filename in enumerate(files):
        old_path = os.path.join(directory, filename)
        new_filename = f"{str(i).zfill(6)}{os.path.splitext(filename)[1]}"
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")


directory_paths = ["/media/pavlos/One Touch/datasets/dynamic_spring_gauss/potato/ims/1",
                   "/media/pavlos/One Touch/datasets/dynamic_spring_gauss/potato/ims/2",
                   "/media/pavlos/One Touch/datasets/dynamic_spring_gauss/potato/seg/0",
                   "/media/pavlos/One Touch/datasets/dynamic_spring_gauss/potato/seg/1",
                   "/media/pavlos/One Touch/datasets/dynamic_spring_gauss/potato/seg/2"]

if __name__ == "__main__":
    for directory in directory_paths:
        number_files(directory)
