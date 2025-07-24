class Field:
    _FIELDS = {
        "A": {
            "WIDTH": 1.8,
            "LENGTH": 2.2,
            "GOAL_WIDTH": 0.4,
            "GOAL_DEPTH": 0.15,
            "CENTER_RADIUS": 0.25,
            "PENALTY_WIDTH": 0.5,
            "PENALTY_DEPTH": 0.15,
            "PENALTY_POINT": 0.375,
            "NUM_ROBOTS": 10,
        },
        "B": {
            "WIDTH": 1.3,
            "LENGTH": 1.5,
            "GOAL_WIDTH": 0.4,
            "GOAL_DEPTH": 0.1,
            "CENTER_RADIUS": 0.2,
            "PENALTY_WIDTH": 0.7,
            "PENALTY_DEPTH": 0.15,
            "PENALTY_POINT": 0.375,
            "NUM_ROBOTS": 6
        }
    }

    def __init__(self, field_type: str):
        if field_type not in self._FIELDS:
            raise ValueError(f"Tipo de campo '{field_type}' inválido. Escolha entre {list(self._FIELDS.keys())}")

        self.WIDTH = self._FIELDS[field_type]["WIDTH"]
        self.LENGTH = self._FIELDS[field_type]["LENGTH"]
        self.GOAL_WIDTH = self._FIELDS[field_type]["GOAL_WIDTH"]
        self.GOAL_DEPTH = self._FIELDS[field_type]["GOAL_DEPTH"]
        self.CENTER_RADIUS = self._FIELDS[field_type]["CENTER_RADIUS"]
        self.PENALTY_WIDTH = self._FIELDS[field_type]["PENALTY_WIDTH"]
        self.PENALTY_DEPTH = self._FIELDS[field_type]["PENALTY_DEPTH"]
        self.PENALTY_POINT = self._FIELDS[field_type]["PENALTY_POINT"]
        self.NUM_ROBOTS = self._FIELDS[field_type]["NUM_ROBOTS"]

    @classmethod
    def from_type(cls, field_type: str):
        """Cria uma instância do campo baseado no tipo escolhido."""
        return cls(field_type)