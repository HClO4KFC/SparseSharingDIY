import os.path

from lxml import etree


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

if __name__ == '__main__':
    labels1 = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5}
    labels2 = {"bus": 6, "car": 7, "cat": 8, "chair": 9, "cow": 10}
    labels3 = {"diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14, "person": 15}
    labels4 = {"pottedplant": 16, "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


    year = '2012'
    txt_name = 'val.txt'
    voc_root = os.path.join('dlfip', 'pascalVOC')
    root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
    # img_root = os.path.join(root, "JPEGImages")
    annotations_root = os.path.join(root, "Annotations")
    output_path_pre = os.path.join(root, "ImageSets", "Main")
    content = [labels1, labels2, labels3, labels4]



    # read train.txt or val.txt file
    txt_path = os.path.join(root, "ImageSets", "Main", txt_name)
    assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

    with open(txt_path) as read:
        xml_list = [os.path.join(annotations_root, line.strip() + ".xml")
                    for line in read.readlines() if len(line.strip()) > 0]
    for i in range(len(content)):
        selected_xml_list = []
        # check file
        idx = 0
        for xml_path in xml_list:

            idx += 1
            if os.path.exists(xml_path) is False:
                print(f"Warning: not found '{xml_path}', skip this annotation file.")
                continue
            # check for targets
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = parse_xml_to_dict(xml)["annotation"]
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue
            flag = False
            for obj in data['object']:
                if obj['name'] in content[i]:
                    flag = True
            if not flag:
                continue
            print(f'{i}:{idx}/{len(xml_list)} ')
            selected_xml_list.append(xml_path)
        selected_xml_list = [path.split('\\')[-1].split('.')[0] for path in selected_xml_list]

        with open(os.path.join(output_path_pre,f'objdet{str(i+1)}_{txt_name}'), 'w') as f:
            for line in selected_xml_list:
                print(line)
                f.write(line.strip()+'\n')
