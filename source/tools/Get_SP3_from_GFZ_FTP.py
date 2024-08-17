#Tools to programatically collect SP3-c orbit files from the GFZ FTP server
import os
import json
import ftplib
from datetime import datetime, timedelta

def load_sp3_codes(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def create_ftp_url(spacecraft, sp3_code, date, orbit_type="RSO"):
    day_str = f"{date.timetuple().tm_yday:03d}"
    year = date.year
    return f"{spacecraft}/ORBIT/{sp3_code}/{orbit_type}/{year}/{day_str}/"

def download_files(ftp_server, path, local_directory):
    try:
        with ftplib.FTP(ftp_server) as ftp:
            ftp.login()
            ftp.cwd(path)
            files = ftp.nlst()
            for filename in files:
                local_path = os.path.join(local_directory, filename)
                with open(local_path, 'wb') as local_file:
                    ftp.retrbinary('RETR ' + filename, local_file.write)
            print(f"Downloaded files to {local_directory}")
    except ftplib.all_errors as e:
        print(f"FTP error: {e}")

def download_sp3(start_date, end_date, spacecraft_name, orbit_type="RSO", json_path="misc/sat_list.json"):
    print(f"Downloading SP3 files for {spacecraft_name} from {start_date} to {end_date}")
    sp3_codes = load_sp3_codes(json_path)
    if spacecraft_name in sp3_codes:
        sp3_code = sp3_codes[spacecraft_name]["sp3-c_code"]
        norad = sp3_codes[spacecraft_name]["norad_id"]
        base_url = "isdcftp.gfz-potsdam.de"
        spacecraft_folder = {
            "CHAMP": "champ",
            "GRACE-FO-A": "grace-fo",
            "GRACE-FO-B": "grace-fo",
            "TerraSAR-X": "tsxtdx",
            "TanDEM-X": "tsxtdx"
        }.get(spacecraft_name, "")
        
        ephemeris_path = f"external/ephems/{spacecraft_name}/NORAD{norad}-{start_date}-{end_date}.txt"
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
    else:
        print(f"No SP3-C code found for {spacecraft_name}")

# If you want to download a specific SP3 file, you can call the download_sp3 function with the desired parameters
# if __name__ == "__main__":
#     print("running from main")
#     start_date = datetime(2023, 4, 23)
#     end_date = datetime(2023, 4, 27)
#     spacecraft_name = "TanDEM-X"
#     download_sp3(start_date, end_date, spacecraft_name)