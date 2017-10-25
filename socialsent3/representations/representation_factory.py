from socialsent3.representations.embedding import SVDEmbedding, Embedding, GigaEmbedding, FullEmbedding
from socialsent3.representations.explicit import Explicit


def create_representation(rep_type, path, *args, **kwargs):
    if rep_type == 'Explicit':
        return Explicit.load(path, *args, **kwargs)
    elif rep_type == 'SVD':
        return SVDEmbedding(path, *args, **kwargs)
    elif rep_type == 'GIGA':
        return GigaEmbedding(path, *args, **kwargs)
    elif rep_type == 'FULL':
        return FullEmbedding(path, *args, *kwargs)
    else:
        return Embedding.load(path, *args, **kwargs)
