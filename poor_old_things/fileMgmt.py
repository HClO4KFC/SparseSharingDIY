import os
import torch
import pickle


def get_data_path(path_pre, dataset, device, step):
    return path_pre + '/parsed_data/', dataset + '_' + device + '_' + str(step) + '.pkl'


def get_model_pre(path_pre, dataset, device, end_num):
    return path_pre + 'trained_models/' + dataset + '_' + device + '_' + str(end_num) + '/'


def save_parsed_data(path_pre, parsed_data, dataset, device, step):
    parsed_data_path_pre, file_name = get_data_path(path_pre=path_pre, dataset=dataset, device=device, step=step)
    if not os.path.exists(parsed_data_path_pre):
        os.makedirs(parsed_data_path_pre)
    print('parsed data saving into', parsed_data_path_pre + file_name + '...')
    with open(parsed_data_path_pre + file_name, "wb") as f:
        pickle.dump(parsed_data, f)


def save_models(path_pre, models, dataset, device, end_num):
    trained_model_path_pre = get_model_pre(path_pre, dataset, device, end_num)
    if not os.path.exists(trained_model_path_pre):
        os.makedirs(trained_model_path_pre)
    for i in range(len(models)):
        base_model = models[i]
        print('tree_list saving into', trained_model_path_pre + 'model_' + str(i) + '.pth...')
        torch.save(base_model, trained_model_path_pre + 'model_' + str(i) + '.pth')


def load_models(path_pre, dataset, device, end_num, ensemble_num, gpu_id):
    trained_model_path_pre = get_model_pre(path_pre, dataset, device, end_num)
    if not os.path.exists(trained_model_path_pre):
        print('trained tree_list are not found in', trained_model_path_pre)
        return None
    models = []
    for i in range(ensemble_num):
        if not os.path.exists(trained_model_path_pre + 'model_' + str(i) + '.pth'):
            print('file not found:', trained_model_path_pre + 'model_' + str(i) + '.pth')
            return None
        base_model = torch.load(trained_model_path_pre + 'model_' + str(i) + '.pth').to('cuda:' + gpu_id)
        base_model.eval()
        models.append(base_model)
    return models


def load_parsed_data(path_pre, dataset, device, step):
    parsed_data_path_pre, file_name = get_data_path(path_pre=path_pre, dataset=dataset, device=device, step=step)
    parsed_data_path = parsed_data_path_pre + file_name
    if not os.path.exists(parsed_data_path):
        print('parsed data is not found:', parsed_data_path)
        return None
    with open(parsed_data_path, 'rb') as f:
        parsed_data = pickle.load(f)
    return parsed_data
