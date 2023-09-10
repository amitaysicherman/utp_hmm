# sbatch --gres=gpu:1,vmem:24g --mem=75G -c4 --time=0-12 --wrap "python bart_gen.py"
import os

from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    BeamSearchScorer,
)
import torch
from cluster_to_phonemes_bart import *
from jiwer import wer


def load_last(model):
    if not os.path.exists(f"models/{config_name}_last.cp"):
        return 0, 1
    checkpoint = torch.load(f"models/{config_name}_last.cp", map_location=device)
    model.load_state_dict(checkpoint['model'])
    load_step = checkpoint['step']
    conf_size = checkpoint['conf_size']
    return load_step, conf_size


# main:
if __name__ == '__main__':
    output_file = "tmp.txt"
    if os.path.exists(output_file):
        os.remove(output_file)

    model = get_model()
    model = model.to(device)

    i, curr_size = load_last(model)
    curr_size = curr_size // 2
    train_dataset = PhonemesDataset(phonemes_file, size=curr_size, samples_count=100)
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    with open(output_file, "a") as f:
        f.write(
            f"load cp-  i:{i}, , curr_size:{curr_size}")
    model = model.train()
    wer_scores = []
    for j, (x_gen, y_ref) in enumerate(train_data):
        x_gen = x_gen.to(device)
        y_ref = y_ref.to(device)
        min_new_tokens = int(0.1 * MAX_LENGTH)
        encoder_input_ids = x_gen
        num_beams = 25
        input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        input_ids = input_ids * PAD_TOKEN

        model_kwargs = {
            "encoder_outputs": model.get_encoder()(
                encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
            )
        }

        beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=num_beams,
            device=model.device,
        )

        logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(min_new_tokens, eos_token_id=END_TOKEN),
            ]
        )
        y_gen = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, max_length=MAX_LENGTH,
                                  **model_kwargs)
        y_gen = [x for x in y_gen[0].cpu().numpy().tolist() if x != PAD_TOKEN]
        y_gen = " ".join([str(x) for x in y_gen])
        # def tensor_to_strings(t):
        #     t = t.cpu().numpy().tolist()
        #     s = " ".join([str(x) for x in t if x not in [PAD_TOKEN, START_TOKEN, END_TOKEN]])
        #     s_list = s.split(str(SEP))
        #     return s_list

        # y_pred = model(input_ids=x_gen, labels=y_ref).logits.argmax(dim=-1)[0]
        # y_pred2 = model(input_ids=torch.ones_like(x_gen) + CLUSTERS_FIRST_TOKEN, labels=y_ref).logits.argmax(dim=-1)[0]
        y_ref = y_ref[0]
        y_ref = [x for x in y_ref.cpu().numpy().tolist() if x != PAD_TOKEN]
        y_ref = " ".join([str(x) for x in y_ref])
        wer_scores.append(wer(y_ref, y_gen))
        with open(output_file, 'a') as f:
            f.write(f"i: {j} , WER : {wer_scores[-1]}\n")
            if j % 100 == 0:
                f.write(f"WER mean : {sum(wer_scores) / len(wer_scores)}\n")

        # with open(output_file, 'a') as f:
        #     f.write(f"i: {i}\n")
        #     f.write(f"y_ref: {y_ref[:20]}\n")
        #     f.write(f"y_pred: {y_pred[:20]}\n")
        #     f.write(f"y_pred2: {y_pred2[:20]}\n")
        #     f.write(f"y_gen: {y_gen[:20]}\n")
        #     f.write("\n")
        # y_gen = tensor_to_strings(y_gen)
        # y_pred = tensor_to_strings(y_pred)
        # y_ref = tensor_to_strings(y_ref)
        # if len(y_gen) != len(y_ref):
        #     y_gen = [y_gen[0]]
        #     y_pred = [y_pred[0]]
        #     y_ref = [y_ref[0]]
        #
        # for y1, y2, y3 in zip(y_ref, y_pred, y_gen):
        #     # wer_score, *vis = compute_wer_and_alignment(y1, y2)
        #
        #     with open(output_file, 'a') as f:
        #         # f.write(f"WER: {wer_score}\n")
        #         f.write(f"y_ref: {y1}\n")
        #         f.write(f"y_pred: {y2}\n")
        #         f.write(f"y_gen: {y3}\n")
        #         f.write("\n")
