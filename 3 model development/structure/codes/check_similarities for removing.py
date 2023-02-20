#https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# sentences = [
#     "Three years later, the coffin was still full of Jello.",
#     "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
#     "The person box was packed with jelly many dozens of months later.",
#     "He found a leprechaun in his walnut shell."
# ]


# model = SentenceTransformer('bert-base-nli-mean-tokens')
# sentence_embeddings = model.encode(sentences)

# print(cosine_similarity(
#     [sentence_embeddings[0]],
#     sentence_embeddings[1:]
# ))
# #manca il download

sentences0 = ["Three years later, the coffin was still full of Jello."]

sentences1 = [
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "He found a leprechaun in his walnut shell."
]


def check_similarity(true_abstract,synthetic_abstract):
    """
    true_abstract -> list 1 value
    synthetic_abstract -> list 1+ values
    """

    true_ab = true_abstract[0]
    print(true_ab)
    print(synthetic_abstract)
    synthetic_abstract.insert(0,true_ab)
    all_ab = synthetic_abstract
    print(all_ab, type(all_ab))
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(all_ab)
    cos = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])
    print(cos, cos.mean())
    return  cos, cos.mean()


check_similarity(sentences0, sentences1)