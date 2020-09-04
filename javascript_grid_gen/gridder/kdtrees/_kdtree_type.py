# coding=utf-8

"""Abstract base super class for acceptable type overrides"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

# from abc import ABC, abstractmethod

class KDTreeType():  # ABC):
	"""
	An abstract base super class (interface) for creating custom
	types that kdtrees can accept. Any custom type must extend
	`KDTreeType` to be acceptable. Based on the purpose of a K-D Tree,
	`KDTreeTypes` must be comparable, indexable, and implement dimensionality
	as well as distance.

	Parameters
	----------
	dim : int
		Dimensionality of this `KDTreeType`.
	"""
	def __init__(self, dim):
		if not isinstance(dim, int):
			raise ValueError("dim must be an int")
		self.dim = dim

	def __eq__(self, other):
		"""
		Determine if `other` is equivalent to this `KDTreeType`.

		Parameters
		----------
		other : object
			The object in question.

		Returns
		-------
		eq : bool
			True if `other` is equivalent to this `KDTreeType`
		"""
		if isinstance(other, self.__class__):
			return self.__dict__ == other.__dict__
		else:
			return False

	def __ne__(self, other):
		"""
		Determine if `other` is not equivalent to this `KDTreeType`.

		Parameters
		----------
		other : object
			The object in question.

		Returns
		-------
		ne : bool
			True if `other` is not equivalent to this `KDTreeType`
		"""
		return not self.__eq__(other)

	# @abstractmethod
	def __lt__(self, other):
		"""
		Return if this `KDTreeType` is 'less' than `other`.

		Parameters
		----------
		other : object
			The object in question.

		Returns
		-------
		lt : bool
			True if this `KDTreeType` is 'less' than `other`.
		"""
		raise NotImplementedError("__lt__ not implemented")

	# @abstractmethod
	def __getitem__(self, key):
		"""
		Return the 'item' and the 'index', `key`, of the `KDTreeType`.
		This needs to be defined based on the custom implementation.

		Returns
		-------
		item : object
			The 'item' and the 'index', `key`, of the `KDTreeType`.
		"""
		raise NotImplementedError("__getitem__ not implemented")

	# @abstractmethod
	def distance(self, other):
		"""
		Calculate the 'distance' between this `KDTreeType` and `other`.

		Parameters
		----------
		other : object
			The object in question.

		Returns
		-------
		dist : float
			'Distance' between this `KDTreeType` and `other`.
		"""
		raise NotImplementedError("distance not implemented")
