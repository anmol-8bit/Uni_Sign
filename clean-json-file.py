import json
import os

# Read the files to delete
files_to_delete = set()
with open('/data_benchmark/Uni-Sign/remove_files_asl.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line.endswith(('.mp4', '.pkl')):
            # Extract just the filename
            filename = os.path.basename(line)
            files_to_delete.add(filename)

# Read and filter JSON
with open('/data_benchmark/Uni-Sign/data/yt_asl_only.cleaned.json', 'r') as f:
    data = json.load(f)

# Filter out entries where video or pose filename matches
filtered_data = []
for entry in data:
    video_filename = entry.get('video', '')
    pose_filename = entry.get('pose', '')
    
    # Keep entry only if neither video nor pose filename is in deletion list
    if video_filename not in files_to_delete and pose_filename not in files_to_delete:
        filtered_data.append(entry)

# Write filtered JSON
with open('/data_benchmark/Uni-Sign/data/filtered_json_file_asl.json', 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"Original entries: {len(data)}")
print(f"Filtered entries: {len(filtered_data)}")
print(f"Removed entries: {len(data) - len(filtered_data)}")