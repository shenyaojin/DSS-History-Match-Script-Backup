import csv
import math
from pathlib import Path


def write_points(path, points):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['point_id', 'x_m', 'y_m'])
        writer.writerows((i, x, y) for i, (x, y) in enumerate(points, 1))


def read_points(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return [(int(row['point_id']), float(row['x_m']), float(row['y_m']))
                for row in csv.DictReader(stream)]


def main():
    directory = Path(__file__).resolve().parent
    before = [(0, 0), (10, 0), (20, 0)]
    after = before.copy()
    after[1] = (after[1][0] + 2, after[1][1])
    write_points(directory / 'before.csv', before)
    write_points(directory / 'after.csv', after)
    original = read_points(directory / 'before.csv')
    modified = read_points(directory / 'after.csv')
    assert original == [(1, 0, 0), (2, 10, 0), (3, 20, 0)]
    assert modified == [(1, 0, 0), (2, 12, 0), (3, 20, 0)]
    distances = [math.hypot(b[1] - a[1], b[2] - a[2])
                 for a, b in zip(original, modified)]
    changed = sum(distance > 0 for distance in distances)
    maximum = max(distances)
    assert changed == 1 and maximum == 2.0
    print(f'changed_points={changed}')
    print(f'maximum_distance_m={maximum:g}')
    print(f'distances_m={distances}')
    print('CSV read-back verification: PASS')
    print('files=before.csv,after.csv,generate_compare.py')


if __name__ == '__main__':
    main()
