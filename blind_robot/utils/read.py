import collections

import matplotlib.pyplot as plt


stoi = {
    "close_drawer": 0,
    "lift_blue_block_drawer": 1,
    "lift_blue_block_slider": 2,
    "lift_blue_block_table": 3,
    "lift_pink_block_drawer": 4,
    "lift_pink_block_slider": 5,
    "lift_pink_block_table": 6,
    "lift_red_block_drawer": 7,
    "lift_red_block_slider": 8,
    "lift_red_block_table": 9,
    "move_slider_left": 10,
    "move_slider_right": 11,
    "open_drawer": 12,
    "place_in_drawer": 13,
    "place_in_slider": 14,
    "push_blue_block_left": 15,
    "push_blue_block_right": 16,
    "push_into_drawer": 17,
    "push_pink_block_left": 18,
    "push_pink_block_right": 19,
    "push_red_block_left": 20,
    "push_red_block_right": 21,
    "rotate_blue_block_left": 22,
    "rotate_blue_block_right": 23,
    "rotate_pink_block_left": 24,
    "rotate_pink_block_right": 25,
    "rotate_red_block_left": 26,
    "rotate_red_block_right": 27,
    "stack_block": 28,
    "turn_off_led": 29,
    "turn_off_lightbulb": 30,
    "turn_on_led": 31,
    "turn_on_lightbulb": 32,
    "unstack_block": 33,
}
itos = {
    0: "close_drawer",
    1: "lift_blue_block_drawer",
    2: "lift_blue_block_slider",
    3: "lift_blue_block_table",
    4: "lift_pink_block_drawer",
    5: "lift_pink_block_slider",
    6: "lift_pink_block_table",
    7: "lift_red_block_drawer",
    8: "lift_red_block_slider",
    9: "lift_red_block_table",
    10: "move_slider_left",
    11: "move_slider_right",
    12: "open_drawer",
    13: "place_in_drawer",
    14: "place_in_slider",
    15: "push_blue_block_left",
    16: "push_blue_block_right",
    17: "push_into_drawer",
    18: "push_pink_block_left",
    19: "push_pink_block_right",
    20: "push_red_block_left",
    21: "push_red_block_right",
    22: "rotate_blue_block_left",
    23: "rotate_blue_block_right",
    24: "rotate_pink_block_left",
    25: "rotate_pink_block_right",
    26: "rotate_red_block_left",
    27: "rotate_red_block_right",
    28: "stack_block",
    29: "turn_off_led",
    30: "turn_off_lightbulb",
    31: "turn_on_led",
    32: "turn_on_lightbulb",
    33: "unstack_block",
}

file = "/Users/emrecanacikgoz/Desktop/rl-project/robot-language/acc.txt"

with open(file, "r", encoding="utf-8") as f:
    lines = f.readlines()

    predictions, targets = [], []
    for line in lines:
        if line.startswith("Preds:"):
            preds_string = line.rstrip("\n").split("Preds: ")[-1]
            preds_list = preds_string.strip("][").split(", ")
            for i in preds_list:
                predictions.append(int(i))

        elif line.startswith("Target:"):
            targets_string = line.rstrip("\n").split("Target: ")[-1]
            targets_list = targets_string.strip("][").split(", ")
            for i in targets_list:
                targets.append(int(i))

        else:
            raise NotImplementedError

true_targetIDs = []
true_predictionIDs = []
wrong_targetIDs = []
wrong_predictionIDs = []
compared = []
for idx, i in enumerate(targets):
    if i != predictions[idx]:
        wrong_targetIDs.append(i)
        wrong_predictionIDs.append(predictions[idx])
        compare = "False: " + itos[i] + " ===> " + itos[predictions[idx]]
        compared.append(compare)
    else:
        true_targetIDs.append(i)
        true_predictionIDs.append(predictions[idx])
        compare = "True: " + itos[i] + " ===> " + itos[predictions[idx]]
        compared.append(compare)

wrong_targets = [itos[i] for i in wrong_targetIDs]
wrong_frequency = dict(collections.Counter(wrong_targets))
wrong_targets_frequency = dict(
    sorted(wrong_frequency.items(), key=lambda item: item[1], reverse=True)
)


true_targets = [itos[i] for i in true_targetIDs]
true_frequency = dict(collections.Counter(true_targets))
true_targets_frequency = dict(
    sorted(true_frequency.items(), key=lambda item: item[1], reverse=True)
)

total_targets = [itos[i] for i in targets]
total_frequency = dict(collections.Counter(total_targets))
total_targets_frequency = dict(
    sorted(total_frequency.items(), key=lambda item: item[1], reverse=True)
)

accuracy = {}
for item in total_targets_frequency.items():
    key, value = item

    if key in total_targets:
        accuracy[key] = str(true_targets_frequency[key]) + "/" + str(value)
    else:
        accuracy[key] = 0

accuracy_frequency = dict(
    sorted(accuracy.items(), key=lambda item: item[1], reverse=True)
)

"""with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(accuracy_frequency, f, ensure_ascii=False, indent=4)"""
plt.hist(wrong_targets, bins=len(set(total_targets)))
plt.xticks(total_targets, rotation=90)
plt.show()
