import os
import fire


def detect_wrong(label_path):
    """
    Detect label files with wrong format: blank file, blank lines, lines with wrong length
    Args:
        label_path: path of label files

    Returns:

    """
    file_list = os.listdir(label_path)
    file_list.sort()
    have_wrong_line = []
    too_much_line = []
    zero_lines = []

    for idx, file in enumerate(file_list):
        print("Checking %d file" % idx)
        with open(str(os.path.join(label_path, file)), 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            zero_lines.append(file)
            continue
        elif len(lines) > 9:
            too_much_line.append(file)
            continue
        else:
            for line in lines:
                label = line.strip().split(' ')
                if len(label) != 9:
                    have_wrong_line.append(file)
                    break
                else:
                    continue

    print("------------------Finished Checking-------------------")
    print("Blank files:")
    print(zero_lines)
    print("Files with too much lines:")
    print(too_much_line)
    print("Files with wrong line:")
    print(have_wrong_line)


if __name__ == "__main__":
    fire.Fire()
