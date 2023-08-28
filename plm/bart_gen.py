#sbatch --gres=gpu:1,vmem:24g --mem=75G -c5 --time=7-0 --wrap "python bart_gen.py"
import os
from cluster_to_phonemes_bart import *
import Levenshtein


def load_last(model):
    if not os.path.exists(f"models/{config_name}_last.cp"):
        return 0, 0, ONE, False, 2

    checkpoint = torch.load(f"models/{config_name}_last.cp", map_location=device)
    model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    load_step = checkpoint['step']
    best_score = checkpoint['best_score']
    conf_type = checkpoint['conf_type']
    conf_dup = checkpoint['conf_dup']
    conf_size = checkpoint['conf_size']
    return load_step, best_score, conf_type, conf_dup, conf_size


def visualize_alignment(reference, hypothesis):
    alignment = Levenshtein.opcodes(reference, hypothesis)
    ref_display = []
    hyp_display = []
    for op, ref_start, ref_end, hyp_start, hyp_end in alignment:
        ref_chunk = reference[ref_start:ref_end]
        hyp_chunk = hypothesis[hyp_start:hyp_end]
        if op == "equal":
            ref_display.append(ref_chunk)
            hyp_display.append(hyp_chunk)
        elif op == "replace":
            max_len = max(len(ref_chunk), len(hyp_chunk))
            ref_display.append('[' + ref_chunk.ljust(max_len) + ']')
            hyp_display.append('[' + hyp_chunk.ljust(max_len) + ']')
        elif op == "delete":
            ref_display.append(ref_chunk)
            hyp_display.append(' ' * (ref_end - ref_start))
        elif op == "insert":
            ref_display.append(' ' * (hyp_end - hyp_start))
            hyp_display.append(hyp_chunk)
    return ''.join(ref_display), ''.join(hyp_display)


# main:
if __name__ == '__main__':
    output_file = "tmp.txt"
    os.remove(output_file)

    model = get_model()
    model = model.to(device)

    i, best_test_acc, curr_type, curr_dup, curr_size = load_last(model)

    train_dataset = PhonemesDataset(phonemes_file, type_=curr_type, dup=curr_dup,
                                    size=curr_size)
    train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(
        f"load cp-  i:{i}, best_test_acc:{best_test_acc}, curr_type:{curr_type}, curr_dup:{curr_dup}, curr_size:{curr_size}")
    model = model.train()

    for j, (x_gen, y_ref) in enumerate(train_data):
        x_gen = x_gen.to(device)

        min_new_tokens = int(0.25 * MAX_LENGTH)

        y_gen = model.generate(x_gen, max_new_tokens=MAX_LENGTH, min_new_tokens=min_new_tokens, num_beams=100)[0]


        def tensor_to_strings(t):
            t = t.cpu().numpy().tolist()
            s = " ".join([str(x) for x in t if x not in [PAD_TOKEN, START_TOKEN, END_TOKEN]])
            s_list = s.split(SEP)
            return s_list


        y_gen = tensor_to_strings(y_gen)
        y_ref = tensor_to_strings(y_ref)
        if len(y_gen) != len(y_ref):
            y_gen = y_gen[0]
            y_ref = y_ref[0]

        for y1, y2 in zip(y_ref, y_gen):
            vis = visualize_alignment(y1, y2)
            with open(output_file, 'a') as f:
                f.write("\n".join(vis) + "\n")
