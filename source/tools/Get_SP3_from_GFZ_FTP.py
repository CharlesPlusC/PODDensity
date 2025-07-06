#Tools to programatically collect SP3-c orbit files from the GFZ FTP server
import os
import json
import ftplib
from datetime import datetime, timedelta

def load_sp3_codes(json_path):
    """
    Load SP3 codes from a JSON file.

    Parameters:
        json_path (str): Path to the JSON file containing SP3 codes.

    Returns:
        dict: Dictionary of SP3 codes loaded from the JSON file.
    """
    with open(json_path, 'r') as file:
        return json.load(file)

def create_ftp_url(spacecraft_folder, sp3_code, date, orbit_type="RSO"):
    """
    Create an FTP URL for accessing SP3 files.

    Parameters:
        spacecraft_folder (str): Folder name for the spacecraft.
        sp3_code (str): SP3 code of the spacecraft.
        date (datetime): Date for which the SP3 file is requested.
        orbit_type (str): Type of orbit (default is "RSO").

    Returns:
        str: The constructed FTP URL path.
    """
    day_str = f"{date.timetuple().tm_yday:03d}"
    year = date.year
    return f"{spacecraft_folder}/ORBIT/{sp3_code}/{orbit_type}/{year}/{day_str}/"

def download_files(ftp_server, path, local_directory):
    """
    Download files from an FTP server to a local directory.

    Parameters:
        ftp_server (str): FTP server address.
        path (str): Path on the FTP server from where to download files.
        local_directory (str): Local directory to save the downloaded files.

    Raises:
        ftplib.all_errors: If any FTP error occurs during the download.
    """
    try:
        with ftplib.FTP(ftp_server) as ftp:
            ftp.login()
            ftp.cwd(path)
            files = ftp.nlst()
            for filename in files:
                local_path = os.path.join(local_directory, filename)
                with open(local_path, 'wb') as local_file:
                    ftp.retrbinary(f'RETR {filename}', local_file.write)
            print(f"Downloaded files to {local_directory}")
    except ftplib.all_errors as e:
        print(f"FTP error: {e}")

def download_sp3(start_date, end_date, spacecraft_name, orbit_type="RSO", json_path="misc/sat_list.json"):
    """
    Download SP3 files for a given spacecraft from the GFZ FTP server within a date range.

    Parameters:
        start_date (datetime): Start date of the period for which to download SP3 files.
        end_date (datetime): End date of the period for which to download SP3 files.
        spacecraft_name (str): Name of the spacecraft.
        orbit_type (str): Type of orbit (default is "RSO").
        json_path (str): Path to the JSON file containing spacecraft SP3 codes (default is "misc/sat_list.json").
    """
    print(f"Downloading SP3 files for {spacecraft_name} from {start_date} to {end_date}")

    sp3_codes = load_sp3_codes(json_path)
    
    if spacecraft_name not in sp3_codes:
        print(f"No SP3-C code found for {spacecraft_name}")
        return

    sp3_code = sp3_codes[spacecraft_name]["sp3-c_code"]
    norad = sp3_codes[spacecraft_name]["norad_id"]
    base_url = "isdcftp.gfz-potsdam.de"

    spacecraft_folder = {
        "CHAMP": "champ",
        "GRACE-FO-A": "grace-fo",
        "GRACE-FO-B": "grace-fo",
        "TerraSAR-X": "tsxtdx",
        "TanDEM-X": "tsxtdx"
    }.get(spacecraft_name)

    if not spacecraft_folder:
        print(f"Spacecraft folder not found for {spacecraft_name}")
        return

    ephemeris_path = f"external/ephems/{spacecraft_name}/NORAD{norad}-{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.txt"

    if os.path.exists(ephemeris_path):
        print(f"Ephemeris file already exists for {spacecraft_name} from {start_date} to {end_date}.")
        return

    current_date = start_date
    while current_date <= end_date:
        ftp_path = create_ftp_url(spacecraft_folder, sp3_code, current_date, orbit_type)
        local_directory = f"external/sp3_files/{sp3_code}/{current_date.year}/{current_date.strftime('%j')}"
        os.makedirs(local_directory, exist_ok=True)
        download_files(base_url, ftp_path, local_directory)
        current_date += timedelta(days=1)

# If you want to download a specific SP3 file, you can call the download_sp3 function with the desired parameters
if __name__ == "__main__":
    print("running from main")
    start_date = datetime(2024, 5, 7)
    end_date = datetime(2024, 5, 16)
    spacecraft_name = "TerraSAR-X"
    download_sp3(start_date, end_date, spacecraft_name)