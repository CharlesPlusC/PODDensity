from ftplib import FTP
import os
import logging

logging.basicConfig(level=logging.INFO)

def download_files_in_directory(ftp, remote_dir, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    try:
        items = ftp.nlst(remote_dir)
    except Exception as e:
        logging.warning(f"Unable to list items in {remote_dir}: {e}")
        return

    for item in items:
        local_path = os.path.join(local_dir, os.path.basename(item))
        if os.path.exists(local_path):
            logging.info(f"Skipping already downloaded: {local_path}")
            continue
        try:
            with open(local_path, 'wb') as file:
                ftp.retrbinary(f'RETR {item}', file.write)
                logging.info(f"Downloaded: {local_path}")
        except Exception as e:
            logging.warning(f"Failed to download {item}: {e}")

ftp = FTP('thermosphere.tudelft.nl')
ftp.login()

base_local_dir = "external/TUDelft_Densities"
directories = [
    # "version_02/GRACE-FO_data",
    "version_02/CHAMP_data"
]

for directory in directories:
    local_dir = os.path.join(base_local_dir, directory.replace('/', '_'))
    download_files_in_directory(ftp, directory, local_dir)

ftp.quit()
