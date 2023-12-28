from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
import numpy as np
import os
import random

target_path = "c-major-scale.mp3"
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


def variation(current_song):
    num = random.randrange(len(names))
    volume = random.randrange(-10, 1)
    start = random.randrange(song_len)
    new_song = construct(current_song, num, volume, start)
    return [mse(song, new_song), [num, volume, start]]


# essentially quiet
base_song = song - 1000
best_mse = 2**100

for i in range(10000):
    variations = [variation(base_song) for _ in range(10)]
    best = sorted(variations, key=lambda a: a[0])[0]
    if best[0] > best_mse:
        continue
    best_mse = best[0]
    print(f"Run: {i}: best: {best}")
    base_song = construct(base_song, *best[1])

base_song.export("approximation-" + target_path)
