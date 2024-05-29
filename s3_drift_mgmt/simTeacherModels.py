def teachers(task_id: int, msg_data: dict, cv_tasks_args):
    # {'input': [sub_out[i] for i in range(len(self.subset_name_list)) if self.subset_name_list[i] == 'rain'][0],
    #  'subsets': sub_out,
    #  'subset_names': self.subset_name_list}
    output_subset_name = cv_tasks_args[task_id].output
    output = [msg_data['subsets'][no] for no in range(len(msg_data['subset_names']))
              if msg_data['subset_names'][no] == output_subset_name][0]
    return output
