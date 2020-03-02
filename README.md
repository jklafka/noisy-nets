# Denoising autoencoders with language models
In this project, we use state-of-the-art linguistic encoders such as BERT and GPT-2 to denoise corrupted text. In this sentence:

"Where is the the teacher?"

a person who speaks English can easily tell that the intended *message* is "where is the teacher". This English speaker unconsciously deletes the repeated "the" from the sentence with minimal, if any, mental effort.

How do language models such as BERT encode noisy sentences like these? Can they help us denoise sentences at scale? What does this process reveal about the model's knowledge of language in general? 
