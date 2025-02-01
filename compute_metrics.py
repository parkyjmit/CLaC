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
    # setting
    model.text_encoder.config.output_hidden_states = True

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

        # decode batch to text
        with torch.no_grad():
            g_feat = model.graph_encoder(graphs)  # (batch, hidden)
            t_feat = model.text_encoder(
                input_ids=questions['input_ids'], 
                attention_mask=questions['attention_mask'], 
                token_type_ids=questions['token_type_ids'],                
            ).hidden_states[-1][:,0]  # (batch, hidden)
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
    parser.add_argument('--data-path', type=str, default='/path/to/example')
    parser.add_argument('--batch-size', type=int, default=1, help='1 for QA')  # 8, 64, 128, 512
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--llm', type=str, default='m3rg-iitd/matscibert')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--label', type=str, default='structure_question_list', choices=['text', 'composition_question_list', 'structure_question_list', 'oxide_question_list'])
    parser.add_argument('--model-ckpt', type=str, default='/path/to/ckpt')      
    

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--evaluation-method', type=str, default='zero-shot QA', choices=['zero-shot QA', 'zero-shot retrieval', 'few-shot QA', 'few-shot retrieval'])

    cfg = parser.parse_args()
    main(cfg)