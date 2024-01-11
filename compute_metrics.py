from data.datamodule import QuestionEvaluationDataModule, CLaMPDataModule
from model.clamp import CLaMPLite
import argparse
from tqdm import tqdm
import torch


def main(cfg):
    print(cfg)
    # DataModule loading
    if cfg.evaluation_method == 'zero-shot QA':
        dm = QuestionEvaluationDataModule(
            data_path=cfg.data_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            tokenizer_model=cfg.llm,
            debug=cfg.debug,
            label=cfg.label
        )
    elif cfg.evaluation_method == 'zero-shot retrieval':
        dm = CLaMPDataModule(
            data_path=cfg.data_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            tokenizer_model=cfg.llm,
            debug=cfg.debug,
            label=cfg.label
        )
    dm.setup()

    # evaluate model
    test_dataloader = dm.test_dataloader()

    # Model loading
    model = CLaMPLite.load_from_checkpoint(cfg.model_ckpt, map_location={'cuda:0': 'cpu'})
    model = model.to(cfg.device)
    model.eval()
    # print vram usage

    # Evaluation
    answer_cnt = 0
    g_stack = []
    t_stack = []
    acc_top1s = []
    acc_top3s = []
    acc_top10s = []
    for batch_i, batch in tqdm(enumerate(test_dataloader)):
        graphs, questions = batch
        graphs = graphs.to(model.device)
        questions = {k: v.to(model.device) for k, v in questions.items()}

        with torch.no_grad():
            g_feat = model.graph_encoder(graphs)  # (batch, hidden)
            t_feat = model.text_encoder(input_ids=questions['input_ids'], attention_mask=questions['attention_mask']).last_hidden_state[:,0]  # (batch, hidden)
            similarity, image_out, text_out = model.loss.global_d(g_feat, t_feat)
            # measure accuracy. the prediction is the largest similarity and the answer is always the first one
            if cfg.evaluation_method == 'zero-shot QA':
                prediction = torch.argmax(similarity, dim=-1)
                if prediction == 0:
                    answer_cnt += 1
            elif cfg.evaluation_method == 'zero-shot retrieval':
                g_stack.append(image_out.cpu())
                t_stack.append(text_out.cpu())
                if batch_i % 8 == 7:
                    g_stack = torch.concat(g_stack, dim=0)
                    t_stack = torch.concat(t_stack, dim=0)
                    # gt_logits = torch.matmul(g_stack, t_stack.transpose(0, 1))
                    gt_logits = torch.matmul(t_stack, g_stack.transpose(0, 1))
                    
                    self_mask = torch.eye(gt_logits.shape[0], device=gt_logits.device, dtype=torch.bool)
                    comb_sim = torch.cat([gt_logits[self_mask][:,None], gt_logits.masked_fill(self_mask, -torch.inf)], dim=-1)
                    sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
                    acc_top1 = (sim_argsort == 0).float().mean()
                    acc_top3 = (sim_argsort < 3).float().mean()
                    acc_top10 = (sim_argsort < 10).float().mean()
                    acc_top1s.append(acc_top1)
                    acc_top3s.append(acc_top3)
                    acc_top10s.append(acc_top10)
                    g_stack = []
                    t_stack = []
    if cfg.evaluation_method == 'zero-shot QA':
        print(f'Accuracy: {answer_cnt / len(test_dataloader)}')
    elif cfg.evaluation_method == 'zero-shot retrieval':
        print(f'Top1: {sum(acc_top1s) / len(acc_top1s)}, Top3: {sum(acc_top3s) / len(acc_top3s)}, Top10: {sum(acc_top10s) / len(acc_top10s)}')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/home/yj/PycharmProjects/MIT/CLaMP/jsons/mp_3d_2020_materials_graphs_gpt_questions')
    # parser.add_argument('--data-path', type=str, default='/home/yj/PycharmProjects/MIT/CLaMP/jsons/mp_3d_2020_nuclear_questions_0')
    # parser.add_argument('--data-path', type=str, default={
    #     'train': '/home/yj/PycharmProjects/MIT/CLaMP/jsons/mp_3d_2020_materials_graphs_gpt_questions_train.parquet',
    #     'val': '/home/yj/PycharmProjects/MIT/CLaMP/jsons/mp_3d_2020_materials_graphs_gpt_questions_val.parquet',
    #     'test': '/home/yj/PycharmProjects/MIT/CLaMP/jsons/mp_3d_2020_materials_graphs_gpt_questions_test.parquet'
    # })
    parser.add_argument('--batch-size', type=int, default=1, help='1 for QA')  # 8, 64, 128, 512
    parser.add_argument('--num-workers', type=int, default=12)
    # parser.add_argument('--llm', type=str, default='allenai/scibert_scivocab_cased')
    parser.add_argument('--llm', type=str, default='m3rg-iitd/matscibert')
    # parser.add_argument('--llm', type=str, default='facebook/galactica-125m')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--label', type=str, default='structure_question_list', choices=['text', 'composition_question_list', 'structure_question_list', 'metal_question_list', 'semiconductor_question_list', 'stable_question_list', 'oxide_question_list', 'Statements'])
    # parser.add_argument('--model-ckpt', type=str, default='outputs/2023-10-22/12-13-19/epoch=35-step=24768.ckpt')  # papers
    # parser.add_argument('--model-ckpt', type=str, default='outputs/2023-10-21/03-47-05/epoch=9-step=1980.ckpt')  # GPT
    # parser.add_argument('--model-ckpt', type=str, default='outputs/2023-11-11/14-27-14/epoch=39-step=27520.ckpt')  # papers-painn
    # parser.add_argument('--model-ckpt', type=str, default='outputs/2023-11-10/11-27-57/epoch=39-step=15800.ckpt')  # gpt-painn
    # parser.add_argument('--model-ckpt', type=str, default='outputs/2023-11-09/00-24-38/epoch=38-step=15405.ckpt')  # gpt-painn-galax
    # parser.add_argument('--model-ckpt', type=str, default='outputs/2023-11-11/04-24-47/epoch=7-step=3160.ckpt')  # gpt-cgcnn-galax
    # parser.add_argument('--model-ckpt', type=str, default='outputs/2023-11-02/13-51-53/epoch=38-step=34515.ckpt')  # merged
    # parser.add_argument('--model-ckpt', type=str, default='outputs/2023-11-19/11-05-21/epoch=87-step=77880.ckpt')  # merged - painn
    # parser.add_argument('--model-ckpt', type=str, default='outputs/2023-12-01/09-27-24/epoch=9-step=1980.ckpt')  # crystal only
    # parser.add_argument('--model-ckpt', type=str, default='outputs/2023-12-01/12-31-20/epoch=19-step=17700.ckpt')  # merged - painn - dlr
    # parser.add_argument('--model-ckpt', type=str, default='outputs/2023-12-02/13-08-19/epoch=28-step=5742.ckpt')  # gpt - painn - dlr
    parser.add_argument('--model-ckpt', type=str, default='outputs/2024-01-11/05-26-26/epoch=38-step=15405.ckpt')  # gpt - painn - matsci - dlr
    
    

    parser.add_argument('--device', type=str, default='cuda:1')

    parser.add_argument('--evaluation-method', type=str, default='zero-shot QA', choices=['zero-shot QA', 'zero-shot retrieval', 'few-shot QA', 'few-shot retrieval'])

    cfg = parser.parse_args()
    main(cfg)