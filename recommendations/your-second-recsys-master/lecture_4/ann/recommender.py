from __future__ import annotations

from itertools import islice
from typing import Callable, Dict, Hashable, Literal, Sequence, Tuple

import numpy as np
from annoy import AnnoyIndex # type: ignore
from numpy.typing import NDArray


class AnnoyRecommender:
    """
    Recommender that uses ndarrays of vectors and/or an Annoy index to yield recommendations

    Attributes
    ----------
    item_vectors
        An array of item embeddings
    user_vectors
        An array of user embeddings
    uid_uiid_mapping
        Mapping from external user ids to internal user ids
    iid_iiid_mapping
        Mapping from external item ids to internal item ids
    uiid_uid_mapping
        Mapping from internal user ids to external user ids
    iiid_iid_mapping
        Mapping from internal item ids to external item ids
    top_k
        Number of recommendations to yield
    dim
        Dimensionality of user/item embeddings
    sim_function
        A callable that computes similarity measure between 2 sets of vectors
    metric
        Annoy metric (for more details visit https://github.com/spotify/annoy)
    n_trees
        Number of trees in annoy index (for more details visit https://github.com/spotify/annoy)
    n_jobs
        Number of cpus Annoy gonna use while building an index (for more details visit https://github.com/spotify/annoy)
    search_k
        Number of tree nodes to inspect (for more details visit https://github.com/spotify/annoy)
    n_neighbors
        Number of neighbors to retrieve from the index (for more details visit https://github.com/spotify/annoy)
    index
        Annoy index
    """
    def __init__(
        self,
        item_vectors: NDArray[np.float32],
        user_vectors: NDArray[np.float32],
        user_id_user_index_id_mapping: Dict[Hashable, int],
        item_id_item_index_id_mapping: Dict[Hashable, int],
        top_k: int,
        dim: int,
        sim_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        metric: Literal['angular', 'euclidian', 'manhattan', 'hamming', 'dot'] = 'dot',
        n_trees: int = 10,
        n_jobs: int = -1,
        search_k: int = -1,
        n_neighbors: int = 500,
    ) -> None:
        self.item_vectors = item_vectors
        self.user_vectors = user_vectors
        self.uid_uiid_mapping = user_id_user_index_id_mapping
        self.iid_iiid_mapping = item_id_item_index_id_mapping
        self.uiid_uid_mapping = {v: k for k, v in user_id_user_index_id_mapping.items()}
        self.iiid_iid_mapping = {v: k for k, v in item_id_item_index_id_mapping.items()}
        self.top_k = top_k
        self.dim = dim
        self.sim_function = sim_function
        self.metric = metric
        self.n_trees = n_trees
        self.n_jobs = n_jobs
        self.search_k = search_k
        self.n_neighbors = n_neighbors

    def fit(self) -> AnnoyRecommender:
        self._build_index()
        return self

    def _build_index(self) -> None:
        index = AnnoyIndex(f=self.dim, metric=self.metric)
        for idx, vector in enumerate(self.item_vectors):
            index.add_item(idx, vector)
        index.build(n_trees=self.n_trees, n_jobs=self.n_jobs)
        self.index = index

    def recommend_single_user(
        self, user_id: Hashable, item_whitelist: Sequence[Hashable]
    ) -> Sequence[Hashable]:
        internal_uid, internal_item_whitelist = self._external_inputs_to_internal(user_id, item_whitelist)
        if len(item_whitelist) == 0:
            internal_item_whitelist = list(self.iid_iiid_mapping.values())
        user_vector = self.user_vectors[internal_uid, :].flatten()

        closest = self._get_similar(user_vector=user_vector)
        closest = self._get_filtered_top(
            candidates=closest, allowed_items=internal_item_whitelist
        )
        return self._map_internal_to_external_id(closest)
    
    def recommend_bruteforce_single_user(self, user_id: Hashable, item_whitelist: Sequence[Hashable]) -> Sequence[Hashable]:
        internal_uid, internal_item_whitelist = self._external_inputs_to_internal(user_id, item_whitelist)
        if len(item_whitelist) == 0:
            internal_item_whitelist = list(self.iid_iiid_mapping.values())
        user_vector = self.user_vectors[internal_uid, :].reshape(1, -1)
        closest = self.sim_function(user_vector, self.item_vectors)
        closest = np.argsort(-closest).flatten().tolist()
        closest = self._get_filtered_top(
            candidates=closest, allowed_items=internal_item_whitelist
        )
        return self._map_internal_to_external_id(closest)

    def _external_inputs_to_internal(self, user_id: Hashable, item_whitelist: Sequence[Hashable]) -> Tuple[int, Sequence[int]]:
        """
        Maps external user id to internal and external item ids to internal ids
        
        Parameters
        ----------
        user_id
            External user id
        item_whitelist
            External ids of items that are allowed to be recommended
        
        Returns
        -------
        internal_uid
            Internal user id
        internal_item_whitelist
            Internal ids of items that are allowed to be recommended
        """
        internal_uid = self.uid_uiid_mapping[user_id]
        internal_item_whitelist = [
            self.iid_iiid_mapping[item] for item in item_whitelist
        ]
        return internal_uid, internal_item_whitelist

    def _get_similar(
        self, user_vector: NDArray[np.float32]
    ) -> Sequence[int]:
        """
        Gets nearest neighbors from an Annoy index

        Parameters
        ----------
        user_vector:
            Numpy array of user's vector representation of shape (1, n),
            where n is the number of dimensions
        
        Returns
        -------
        nearest_negihbors
            A sequence of sorted similar items to a given user_vector
        """
        nearest_neighbors = self.index.get_nns_by_vector(
            user_vector,
            self.n_neighbors,
            search_k=self.search_k,
            include_distances=False,
        )
        return nearest_neighbors

    def _get_filtered_top(
        self, candidates: Sequence[int], allowed_items: Sequence[int]
    ) -> Sequence[int]:
        """
        Takes candidates, intersects with allowed items and returns top_k similar items

        Parameters
        ----------
        candidates:
            A sequence of candidates to recommend
        allowed_items:
            A sequence of items allowed to recommend
        
        Returns
        -------
        A sequence of filtered top_k recommendations
        """
        allowed_items_set = set(allowed_items)
        return list(
            islice(
                (cand for cand in candidates if cand in allowed_items_set), self.top_k
            )
        )

    def _map_internal_to_external_id(
        self, seq_to_map: Sequence[int]
    ) -> Sequence[Hashable]:
        return [self.iiid_iid_mapping[item] for item in seq_to_map]
