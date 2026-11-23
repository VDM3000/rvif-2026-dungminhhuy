# Early Fusion 3/1 comparison

The four predefined EMS validation folds contain 40 people each. MSNet uses the historical pretrained checkpoint for each fold; the new early-fusion head is refitted on the other 120 people after choosing its epoch inside those 120. The validation-score Otsu rule matches the historical notebook and does not use labels. The historical late-fusion configuration was selected using these same development folds, so this comparison is descriptive and is not independent validation. The separate 2/1/1 experiment has different training-set sizes and threshold rules.
