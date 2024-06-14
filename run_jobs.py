import subprocess
import re
from tqdm import tqdm

# List of bash commands to be executed
commands = [
    # 'python train.py -d /home/pavlos/Desktop/data/our_baseline -t 2 -i 10000 -r 1 -o /home/pavlos/Desktop/output/gaussian_assets_output -s whale_white',
    # 'python train.py -d /home/pavlos/Desktop/data/our_baseline -t 2 -i 10000 -r 1 -o /home/pavlos/Desktop/output/gaussian_assets_output -s torus_white',
    # 'python train.py -d /home/pavlos/Desktop/data/our_baseline -t 2 -i 10000 -r 1 -o /home/pavlos/Desktop/output/gaussian_assets_output -s cow_white',
    # 'python train.py -d /home/pavlos/Desktop/data/our_baseline -t 2 -i 10000 -r 1 -o /home/pavlos/Desktop/output/gaussian_assets_output -s ball_white',
    # 'python train.py -d /home/pavlos/Desktop/data/dynamic_pac -t 2 -i 10000 -r 1 -o /home/pavlos/Desktop/output/gaussian_assets_output -s elastic_0',
    'python train.py -d /home/pavlos/Desktop/data/dynamic_pac -t 2 -i 10000 -r 1 -o /home/pavlos/Desktop/output/gaussian_assets_output -s elastic_1',
    # 'python train.py -d /home/pavlos/Desktop/data/dynamic_pac -t 2 -i 10000 -r 1 -o /home/pavlos/Desktop/output/gaussian_assets_output -s elastic_2',
    # 'python train.py -d /home/pavlos/Desktop/data/dynamic_pac -t 2 -i 10000 -r 1 -o /home/pavlos/Desktop/output/gaussian_assets_output -s elastic_3',
    # 'python train.py -d /home/pavlos/Desktop/data/dynamic_pac -t 2 -i 10000 -r 1 -o /home/pavlos/Desktop/output/gaussian_assets_output -s elastic_4',
    # 'python train.py -d /home/pavlos/Desktop/data/dynamic_pac -t 2 -i 10000 -r 1 -o /home/pavlos/Desktop/output/gaussian_assets_output -s elastic_5',
    'python train.py -d /home/pavlos/Desktop/data/dynamic_pac -t 2 -i 10000 -r 1 -o /home/pavlos/Desktop/output/gaussian_assets_output -s elastic_6',
    'python train.py -d /home/pavlos/Desktop/data/dynamic_pac -t 2 -i 10000 -r 1 -o /home/pavlos/Desktop/output/gaussian_assets_output -s elastic_7',
    # 'python train.py -d /home/pavlos/Desktop/data/dynamic_pac -t 2 -i 10000 -r 1 -o /home/pavlos/Desktop/output/gaussian_assets_output -s elastic_8',
    # 'python train.py -d /home/pavlos/Desktop/data/dynamic_pac -t 2 -i 10000 -r 1 -o /home/pavlos/Desktop/output/gaussian_assets_output -s elastic_9'
]


# Function to create a safe filename from a command
def safe_filename(command):
    # Remove characters that are not alphanumeric or spaces
    filename = re.sub(r'[^a-zA-Z0-9 ]', '', command)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    return filename


# Execute each command serially and write output to a separate file
for i, command in enumerate(tqdm(commands), 1):
    try:
        # Execute the command and capture the output
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.CalledProcessError as e:
        stdout = e.stdout
        stderr = e.stderr
        error = str(e)

    # Create a safe filename for the output file
    filename = safe_filename(command)
    output_filename = f"job_output/{i}_{filename}.txt"

    # Write the output to the file
    with open(output_filename, 'w') as file:
        file.write(f"Command: {command}\n")
        file.write(f"Standard Output:\n{stdout}\n")
        file.write(f"Standard Error:\n{stderr}\n")
        if 'error' in locals():
            file.write(f"Error: {error}\n")
        file.write("=" * 50 + "\n")
