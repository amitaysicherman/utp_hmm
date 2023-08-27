from cluster_to_phonemes_bart import *


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


# main:
if __name__ == '__main__':

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

        y_gen = model.generate(x_gen, max_new_tokens=MAX_LENGTH, min_new_tokens=min_new_tokens, num_beams=100)

        y_gen = y_gen.cpu().numpy().tolist()
        y_gen = " ".join([str(x) for x in y_gen if x not in [PAD_TOKEN, START_TOKEN, END_TOKEN, SEP]])

        y_ref = y_ref[0].cpu().numpy().tolist()
        y_ref = " ".join([str(x) for x in y_ref if x not in [PAD_TOKEN, START_TOKEN, END_TOKEN, SEP]])
        print(
            f'x: {" ".join([str(x) for x in x_gen[0].cpu().numpy().tolist() if x not in [PAD_TOKEN, START_TOKEN, END_TOKEN, SEP]])}\n')
        print(f'gen: {y_gen}\n')
        print(f'ref: {y_ref}\n\n')
