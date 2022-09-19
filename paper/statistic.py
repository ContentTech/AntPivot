import pickle
import matplotlib.pyplot as plt
import numpy as np


with open("statistic.pkl", "rb") as f:
    stat = pickle.load(f)

 # results = {
 #        "total_duration": entire_time, *
 #        "highlight_ratio": highlight_ratio, *
 #        "highlight_time": highlight_time, *
 #        "highlight_num": highlight_num,
 #        "utterance_ratio": talk_ratio, *
 #        "utterance_time": all_sent_time, *
 #        "utterance_num": sent_num,
 #        "word_per_utterance": all_word_num,
 #        "utterance_per_highlight": uph


# total_duration = np.array(stat["total_duration"]).clip(min=200)
# plt.hist(total_duration, bins=30, histtype='bar', color='#f3722c')
# plt.xlabel("Duration (s)")
# plt.savefig("total_duration.svg")

# utterance_num = np.array(stat["utterance_num"])
# plt.hist(utterance_num, bins=30, histtype='bar', color='#f8961e')
# plt.xlabel("Utterance number per record")
# plt.savefig("utterance_num.svg")

# utterance_per_highlight = np.array(stat["utterance_per_highlight"]).clip(max=110)
# plt.hist(utterance_per_highlight, bins=30, histtype='bar', color='#f9844a')
# plt.xlabel("Utterance number per highlight")
# plt.savefig("utterance_per_highlight.svg")

# word_per_utterance = np.array(stat["word_per_utterance"]).clip(max=125)
# plt.hist(word_per_utterance, bins=30, histtype='bar', color='#f9c74f')
# plt.xlabel("Word number per utterance")
# plt.savefig("word_per_utterance.svg")

# highlight_num = np.array(stat["highlight_num"]).clip(max=36)
# plt.hist(highlight_num, bins=30, histtype='bar', color='#90be6d')
# plt.xlabel("Highlight number per record")
# plt.savefig("highlight_num.svg")

# highlight_time = np.array(stat["highlight_time"])
# plt.hist(highlight_time, bins=30, histtype='bar', color='#43aa8b')
# plt.xlabel("Duration (s)")
# plt.savefig("highlight_time.svg")


# utterance_time = np.array(stat["utterance_time"]).clip(max=55)
# plt.hist(utterance_time, bins=30, histtype='bar', color='#f8961e')
# plt.xlabel("Duration (s)")
# plt.savefig("utterance_time.png")



# utterance_ratio = np.array(stat["utterance_ratio"]).clip(min=0.45)
# plt.hist(utterance_ratio, bins=30, histtype='bar', color='#4d908e')
# plt.xlabel("Ratio")
# plt.savefig("utterance_ratio.svg")

highlight_ratio = np.array(stat["highlight_ratio"]).clip(max=0.35)
plt.hist(highlight_ratio, bins=30, histtype='bar', color='#90be6d')
plt.xlabel("Ratio")
plt.savefig("highlight_ratio.svg")












plt.ylabel("Item Number")
plt.show()
