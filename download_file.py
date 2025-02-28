import gdown

# Google Drive file ID (extracted from the link)
file_id = "1eAZUQLfzxBtWqLr9qx845NRJTw2kM0Pn"

# Construct the download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Output file name
output = "users-score-2023.csv"

# Download the file
gdown.download(url, output, quiet=False)

print(f"File downloaded and saved as {output}")
