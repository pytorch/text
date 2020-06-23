import glob
import json
import os
import subprocess
from tqdm import tqdm


def output_checksums_to_files(dir=".checksums"):
    if not os.path.exists(dir):
        os.makedirs(dir)

    processes = []
    with open("languages_fast_text.txt", 'r') as f:
        num_languages = 0
        for line in f:
            num_languages += 1
            language = line.strip()
            filepath = '{}/{}.txt'.format(dir, language)
            url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'.format(language)
            processes.append(subprocess.Popen(['./get_checksum.sh', filepath, url]))

        print('Computing checksums')
        with tqdm(unit_scale=0, unit='files', total=num_languages) as t:
            for p in processes:
                p.wait()
                t.update(1)


def process_checksums_to_json_file(dir=".checksums"):
    if not os.path.exists(dir):
        os.makedirs(dir)
    os.chdir(dir)

    checksums = {}
    for file_name in glob.glob("*.txt"):
        file_base_name = os.path.splitext(file_name)[0]
        url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'.format(file_base_name)

        with open(file_name, 'r') as f:
            sha256hash = f.readline()
            checksums[url] = sha256hash
    checksums_json = json.dumps(checksums)

    with open("checksums_fast_text.json", 'w') as f:
        f.write(checksums_json)


def main():
    dir = ".checksums"
    json_file_path = os.path.join(os.getcwd(), dir, "checksums_fast_text.json")
    output_checksums_to_files(dir=dir)
    process_checksums_to_json_file(dir=dir)

    print("Path to FastTest checksum file: {}".format(json_file_path))


if __name__ == "__main__":
    main()
