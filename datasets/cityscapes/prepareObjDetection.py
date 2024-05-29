import glob
import json
import os.path


def gen(train_val: str):
    # person_path_like = os.path.join(person_path_pre, train_val, '**', '*_gtBboxCityPersons.json')
    person_path_like = os.path.join('D:\\doing\\cvDatasets\\cityscapes\\gtBbox_cityPersons', train_val, '**', '*_gtBboxCityPersons.json')
    car_path_like = os.path.join('D:\\doing\\cvDatasets\\cityscapes\\gtBbox3d', train_val, '**', '*_gtBbox3d.json')
    person_file_list = glob.glob(person_path_like, recursive=True)
    car_file_list = glob.glob(car_path_like, recursive=True)
    person_file_list.sort()
    car_file_list.sort()

    assert len(person_file_list) == len(car_file_list)
    for i in range(len(person_file_list)):
        person_file = person_file_list[i]
        car_file = car_file_list[i]
        with open(person_file, 'r') as f:
            person_data = json.load(f)
        bbox_per = []
        label_per = []
        for dat_obj in person_data['objects']:
            new_bbox = dat_obj['bbox']
            new_label = dat_obj['label']
            bbox_per.append(new_bbox)
            label_per.append(new_label)
        with open(car_file, 'r') as f:
            car_data = json.load(f)
        bbox_car = []
        label_car = []
        for dat_obj in car_data['objects']:
            new_bbox = dat_obj['2d']['modal']
            new_label = dat_obj['label']
            bbox_car.append(new_bbox)
            label_car.append(new_label)
        bboxs = bbox_car + bbox_per
        labels = label_car + label_per
        objs = []
        for bbox, label in zip(bboxs, labels):
            objs.append({
                'bbox': bbox,
                'label': label
            })
        output_file = {
            'imgWidth': car_data['imgWidth'],
            'imgHeight': car_data['imgHeight'],
            'objects': objs
        }
        # output_file_json_pre = os.path.join(person_path_pre.replace('gtBbox_cityPersons', 'gtBbox_objects'), train_val, person_file.replace('_gtBboxCityPersons', '_gtBboxObjects'))
        output_file_path = car_file.replace('gtBbox3d', 'gtBbox_obj')
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)
            os.rmdir(output_file_path)
        with open(output_file_path, 'w') as f:
            json.dump(output_file, f, indent=4)
            print('writing into ' + output_file_path + ' ...')


if __name__ == '__main__':
    gen('train')
    gen('val')
