from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
import numpy as np
import os
import random

target_path = "c-major-scale.mp3"
out_path = "approximation-" + target_path
sfx_path = "sfx/"

song = AudioSegment.from_file(file=target_path)
song_len = len(song)

sfx_files = os.listdir(sfx_path)
names = [f"{sfx_path}{file}" for file in sfx_files]

print("Loading sfx:")

files = []
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for name in names:
        futures.append(executor.submit(AudioSegment.from_file, name))
    for future in futures:
        files.append(future.result())


def mse(song1, song2):
    array1 = np.array(song1.get_array_of_samples())
    array2 = np.array(song2.get_array_of_samples())

    diff = len(array1) - len(array2)
    if diff > 0:
        array1 = array1[:-diff]
    elif diff < 0:
        array2 = array2[:diff]
    m = np.sum(np.abs(array1 - array2))
    return m


def construct(song, num, volume, start):
    sfx = files[num]
    sfx += volume
    sfx = song.overlay(sfx, position=start)
    return sfx


def full_construct(variations):
    new_song = song - 10000
    for variation in variations:
        new_song = construct(new_song, *variation)
    return new_song


def variation(current_song):
    num = random.randrange(len(names))
    volume = random.randrange(-10, 1)
    start = random.randrange(song_len)
    new_song = construct(current_song, num, volume, start)
    return [mse(song, new_song), [num, volume, start]]


def inverse(sfx):
    frame_rate = sfx.frame_rate
    sample_width = sfx.sample_width
    channels = sfx.channels

    array = np.array(sfx.get_array_of_samples())
    array = np.negative(array)
    raw_data = array.tobytes()

    return AudioSegment(
        data=raw_data,
        sample_width=sample_width,
        frame_rate=frame_rate,
        channels=channels,
    )


def filter(best_mce, best_song, inverses):
    if len(inverses) == 0:
        return []

    bad = []
    for index, inverse in enumerate(inverses):
        test = best_song.overlay(inverse)
        diff = mse(song, test)
        if diff < best_mce:
            bad.append(index)

    return bad


# essentially quiet
quiet = song - 10000
base_song = song - 10000
best_mse = 2**100
best_v = []
inverses = []

for i in range(1000000):
    if i % 10000 == 0 and i != 0:
        print(f"Run: {i}: best: {best}")
        bad = filter(best_mse, base_song, inverses)
        best_v = [v for i, v in enumerate(best_v) if i not in bad]
        inverses = [v for i, v in enumerate(inverses) if i not in bad]
        base_song = full_construct(best_v)
        best_mse = mse(song, base_song)
        base_song.export(out_path)

    variations = [variation(base_song) for _ in range(10)]
    best = sorted(variations, key=lambda a: a[0])[0]
    if best[0] > best_mse:
        continue
    best_mse = best[0]
    best_v.append(best[1])
    just_sound = construct(quiet, *best[1])
    inverses.append(inverse(just_sound))
    base_song = base_song.overlay(just_sound)

base_song.export(out_path)
