import torch
import os
import numpy as np

from utils.dataParse import ParsedDataset, data_slice


def eval_and_test(ensemble_list: list, parsed_data: ParsedDataset,
                  batch_size: int, device: str, is_last_iter: bool,
                  save_path_pre: str, end_num: int) -> tuple[
                    torch.Tensor, torch.Tensor, torch.Tensor,
                    torch.Tensor, torch.Tensor, torch.Tensor,
                    list, list, any, any]:
    x_eval, y_eval, mask_eval = parsed_data.get_eval_set()
    x_test, y_test, mask_test = parsed_data.get_test_set()
    ensemble_eval_output = torch.Tensor([])
    ensemble_test_output = torch.Tensor([])
    ensemble_eval_gt = torch.Tensor([])
    ensemble_test_gt = torch.Tensor([])
    ensemble_eval_mask = torch.Tensor([])
    ensemble_test_mask = torch.Tensor([])
    ensemble_task_embedding_list = []
    ensemble_encoder_output_list = []
    # 验证各基模型在验证集上的表现
    for base_no in range(len(ensemble_list)):
        eval_batch_num = int(len(x_eval) / batch_size)
        print('evaluating base model no.', base_no, ', sized', batch_size, 'x', eval_batch_num, '.')
        base_model = ensemble_list[base_no]
        base_eval_output = torch.Tensor([])
        base_eval_gt = torch.Tensor([])
        base_eval_mask = torch.Tensor([])
        # 分批验证
        for batch in range(eval_batch_num + 1):
            print('evaluating batch', batch)

            # 切出本批次的x,y和mask
            batch_x, batch_y, batch_mask = data_slice(*(parsed_data.get_eval_set()), batch, batch_size)
            task_id_batch = parsed_data.task_id_repeated[:len(batch_x)].to(device)
            if len(batch_x) == 0:
                return

            # 跑一批验证集数据
            output, attentions, task_embedding, encoder_output = base_model(batch_x, task_id_batch)
            if base_no == 0:
                encoder_output = torch.mul(encoder_output.permute(2, 0, 1), batch_x).permute(1, 2, 0)
                if is_last_iter:
                    ensemble_task_embedding_list.append(task_embedding.cpu().detach().numpy())
                    ensemble_encoder_output_list.append(encoder_output.cpu().detach().numpy())
                    if not os.path.exists('./savings/embedding_collect/' + save_path_pre + '/'):
                        os.mkdir('./savings/embedding_collect/' + save_path_pre + '/')
                    np.save('./savings/embedding_collect/' + save_path_pre +
                            '/task_embedding_list_eval_' + str(base_no) + '_' + str(end_num))
                    np.save('./savings/embedding_collect/' + save_path_pre +
                            '/encoder_output_list_eval_' + str(base_no) + '_' + str(end_num))

            output = output.mul(batch_mask)
            base_eval_output = torch.cat([base_eval_output, output.cpu().detach()], 0)
            base_eval_gt = torch.cat([base_eval_gt, batch_y.cpu().detach()], 0)
            base_eval_mask = torch.cat([base_eval_mask, batch_mask.cpu().detach()], 0)

        # 将所有基模型的eval阶段output和mask汇总到ensemble
        if len(ensemble_eval_output) == 0:
            ensemble_eval_output = base_eval_output.clone().unsqeeze(dim=0)
        else:
            ensemble_eval_output = torch.cat([ensemble_eval_output, base_eval_output.unsqueeze(dim=0)])
        ensemble_eval_mask = torch.cat([ensemble_eval_mask, base_eval_mask.cpu().detach()], 0)
        ensemble_eval_gt = torch.cat([ensemble_eval_gt, base_eval_gt])

        # test阶段:
        base_test_output = torch.Tensor([])
        base_test_gt = torch.Tensor([])
        base_test_mask = torch.Tensor([])
        test_batch_num = int(len(x_test) / batch_size)
        print('testing base model no.', base_no, ', sized', batch_size, 'x', test_batch_num, '.')
        for batch in range(+1):
            print('testing batch', batch)
            batch_x, batch_y, batch_mask = data_slice(x_test, y_test, mask_test, batch, batch_size)
            if len(batch_x) == 0:
                break
            # 用训练好的基模型跑一批test数据
            task_id_batch = parsed_data.task_id_repeated[:len(batch_x)].to(device)
            output, attentions, _, _ = base_model(batch_x, task_id_batch)
            output = output.mul(batch_mask)
            # 保存测试阶段该基模型在所有测试输入上的输出,gt和mask
            base_test_output = torch.cat([base_eval_output, output.cpu().detach()], 0)
            base_test_gt = torch.cat([base_test_gt, batch_y.cpu().detach()], 0)
            base_test_mask = torch.cat([base_test_mask, batch_mask.cpu().detach()], 0)

        # 将所有基模型在测试阶段的输出数据集中到ensemble
        if len(ensemble_test_output) == 0:
            ensemble_test_output = base_test_output.clone().unsqeeze(dim=0)
        else:
            ensemble_test_output = torch.cat([ensemble_test_output, base_test_output])
        ensemble_test_mask = torch.cat([ensemble_test_mask, base_test_mask])
        ensemble_test_gt = torch.cat([ensemble_test_gt, base_test_gt])
    # 总结:至此已经完成了本次epoch的分批训练和分批验证,接下来截止到本epoch,组成的集成模型loss情况
    return ensemble_eval_output, ensemble_eval_gt, ensemble_eval_mask, \
        ensemble_test_output, ensemble_test_gt, ensemble_test_mask, \
        ensemble_encoder_output_list, ensemble_task_embedding_list, \
        x_eval, x_test
