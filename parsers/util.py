import os
import zipfile


def zip_files(src_dir, zip_name, file_types_to_zip):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if any(file.endswith(ext) for ext in file_types_to_zip):
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), src_dir))


def remove_files_by_extension(src_dir, extension):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(extension):
                os.remove(os.path.join(root, file))
