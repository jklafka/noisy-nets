import torch, argparse, csv
from transformers import pipeline

# device = torch.device("cuda:0")
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("output_file")
args = parser.parse_args()


pairs = []
with open("Stimuli/" + args.input_file + ".csv", 'r') as input_file:
    reader = csv.reader(input_file)
    for row in reader:
        pairs.append(row)

# initialize lm encoder
lm = pipeline("feature-extraction", model = "bert-base-uncased",
                device=-1)

# ------ encoding --------
for pair in pairs:
    noisy_text = pair[0]
    clean_text = pair[1]


    with torch.no_grad():
        encoded_clean_text = lm(clean_text)
        encoded_noisy_text = lm(noisy_text)

    with open("Vectors/" + args.output_file + ".csv", 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(encoded_clean_text[0][-1])
        writer.writerow(encoded_noisy_text[0][-1]) 
