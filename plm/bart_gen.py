# sbatch --gres=gpu:1,vmem:24g --mem=75G -c4 --time=0-12 --wrap "python bart_gen.py"
import os

import torch

from cluster_to_phonemes_bart import *
import Levenshtein


def load_last(model):
    if not os.path.exists(f"models/{config_name}_last.cp"):
        return 0, 0, 1
    checkpoint = torch.load(f"models/{config_name}_last.cp", map_location=device)
    model.load_state_dict(checkpoint['model'])
    load_step = checkpoint['step']
    best_score = checkpoint['best_score']
    conf_size = checkpoint['conf_size']
    return load_step, best_score, conf_size


def compute_wer_and_alignment(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not len(ref_words) or not len(hyp_words):
        return len(hyp_words), '', ' '.join(['[ ]'] * len(hyp_words))
    alignment = Levenshtein.opcodes(ref_words, hyp_words)

    ref_display = []
    hyp_display = []

    S, D, I = 0, 0, 0  # Counters for substitutions, deletions, and insertions

    for op, ref_start, ref_end, hyp_start, hyp_end in alignment:
        ref_chunk = ref_words[ref_start:ref_end]
        hyp_chunk = hyp_words[hyp_start:hyp_end]

        if op == "equal":
            ref_display.extend(ref_chunk)
            hyp_display.extend(hyp_chunk)
        elif op == "replace":
            max_len = max(len(ref_chunk), len(hyp_chunk))
            S += max_len
            ref_display.extend(['[' + word + ']' for word in ref_chunk] + ['[ ]'] * (max_len - len(ref_chunk)))
            hyp_display.extend(['[' + word + ']' for word in hyp_chunk] + ['[ ]'] * (max_len - len(hyp_chunk)))
        elif op == "delete":
            D += (ref_end - ref_start)
            ref_display.extend(ref_chunk)
            hyp_display.extend(['[ ]'] * (ref_end - ref_start))
        elif op == "insert":
            I += (hyp_end - hyp_start)
            ref_display.extend(['[ ]'] * (hyp_end - hyp_start))
            hyp_display.extend(hyp_chunk)

    wer_score = (S + D + I) / len(ref_words)
    return wer_score, ' '.join(ref_display), ' '.join(hyp_display)


# main:
if __name__ == '__main__':
    output_file = "tmp2.txt"
    if os.path.exists(output_file):
        os.remove(output_file)

    model = get_model()
    model = model.to(device)

    i, best_score, curr_size = load_last(model)
    curr_size = curr_size // 2
    train_dataset = PhonemesDataset(phonemes_file, size=curr_size, samples_count=100)
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    with open(output_file, "a") as f:
        f.write(
            f"load cp-  i:{i}, best_test_acc:{best_score}, curr_size:{curr_size}")
    model = model.train()

    for j, (x_gen, y_ref) in enumerate(train_data):
        x_gen = x_gen.to(device)
        y_ref = y_ref.to(device)
        min_new_tokens = int(0.25 * MAX_LENGTH)

        from transformers import (
            LogitsProcessorList,
            MinLengthLogitsProcessor,
            BeamSearchScorer,
        )
        import torch

        encoder_input_ids = x_gen

        num_beams = 6
        input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        input_ids = input_ids * START_TOKEN

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
        y_gen2 = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor,max_length=MAX_LENGTH, **model_kwargs)

        y_gen = model.generate(x_gen[:, :-1], max_new_tokens=MAX_LENGTH, min_new_tokens=min_new_tokens, num_beams=num_beams,
                               decoder_start_token_id=END_TOKEN)[0]
        with open(output_file, 'a') as f:
            f.write('y_gen', y_gen)
            f.write('y_gen2', y_gen2)


        def tensor_to_strings(t):
            t = t.cpu().numpy().tolist()
            s = " ".join([str(x) for x in t if x not in [PAD_TOKEN, START_TOKEN, END_TOKEN]])
            s_list = s.split(str(SEP))
            return s_list


        y_pred = model(input_ids=x_gen, labels=y_ref).logits.argmax(dim=-1)[0]
        y_pred2 = model(input_ids=torch.ones_like(x_gen) + CLUSTERS_FIRST_TOKEN, labels=y_ref).logits.argmax(dim=-1)[0]
        y_ref = y_ref[0]
        with open(output_file, 'a') as f:
            f.write(f"i: {i}\n")
            f.write(f"y_ref: {y_ref[:20]}\n")
            f.write(f"y_pred: {y_pred[:20]}\n")
            f.write(f"y_pred2: {y_pred2[:20]}\n")
            f.write(f"y_gen: {y_gen[:20]}\n")
            f.write("\n")
        y_gen = tensor_to_strings(y_gen)
        y_pred = tensor_to_strings(y_pred)
        y_ref = tensor_to_strings(y_ref)
        if len(y_gen) != len(y_ref):
            y_gen = [y_gen[0]]
            y_pred = [y_pred[0]]
            y_ref = [y_ref[0]]

        for y1, y2, y3 in zip(y_ref, y_pred, y_gen):
            # wer_score, *vis = compute_wer_and_alignment(y1, y2)

            with open(output_file, 'a') as f:
                # f.write(f"WER: {wer_score}\n")
                f.write(f"y_ref: {y1}\n")
                f.write(f"y_pred: {y2}\n")
                f.write(f"y_gen: {y3}\n")
                f.write("\n")
