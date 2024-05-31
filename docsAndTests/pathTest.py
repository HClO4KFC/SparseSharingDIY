import os

def split_path(path):
    parts = []
    while True:
        path, tail = os.path.split(path)
        if tail:
            parts.append(tail)
        else:
            if path:
                parts.append(path)
            break
    parts.reverse()
    return parts

# 示例路径
path = "..\\cvDatasets/a/s/d\\f\\e.cpp"
print(split_path(path))
print(path)