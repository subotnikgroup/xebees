    def _make_hash(self):
        def recursive_hash(obj):
            if isinstance(obj, xp.ndarray):
                if obj.flags.writeable:
                    raise ValueError("Refusing to hash mutable array")
                return hash((obj.shape, obj.dtype, obj.tobytes()))
            elif isinstance(obj, tuple):
                return hash(tuple(recursive_hash(x) for x in obj))
            else:
                return hash(obj)

        return reduce(
            operator.xor,
            (recursive_hash(getattr(self, key)) for key in self.__slots__ if key not in ['_locked', '_hash']),
            0)
