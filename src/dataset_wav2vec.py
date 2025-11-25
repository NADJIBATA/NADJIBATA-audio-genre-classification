import os
import soundfile as sf
import random

def create_wav2vec_manifests(root_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    files = []
    for genre in sorted(os.listdir(root_dir)):
        genre_path = os.path.join(root_dir, genre)
        if not os.path.isdir(genre_path):
            continue

        for f in os.listdir(genre_path):
            if f.endswith(".wav"):
                files.append(f"{genre}/{f}")

    random.seed(42)
    random.shuffle(files)

    n = len(files)
    train = files[:int(0.7*n)]
    valid = files[int(0.7*n):int(0.85*n)]
    test  = files[int(0.85*n):]

    LABELS = sorted(os.listdir(root_dir))
    LABEL_MAP = {g:i for i,g in enumerate(LABELS)}

    def write_manifest(split, data):
        with open(f"{out_dir}/{split}.tsv", "w") as f:
            f.write(root_dir + "\n")
            for item in data:
                wav_path = os.path.join(root_dir, item)
                info = sf.info(wav_path)
                nsamples = int(info.samplerate * info.duration)
                f.write(f"{item}\t{nsamples}\n")

        with open(f"{out_dir}/{split}.lbl", "w") as f:
            for item in data:
                genre = item.split("/")[0]
                f.write(str(LABEL_MAP[genre]) + "\n")

    write_manifest("train", train)
    write_manifest("valid", valid)
    write_manifest("test",  test)

    print("Manifests created in", out_dir)
